export RemoteEstObj,
    LocalEstObj,
    EstimationWrapper,
    update_reo!,
    reset_reo!,
    theta0, theta1, hess, invhess, grad,
    getworkers,
    parallel_simloglik!,
    update!,
    start_up_workers

"workers() but excluding master"
getworkers() = filter(i -> i != 1, workers())

function start_up_workers(ENV::Base.EnvDict)
    # send primitives to workers
    oldworkers = getworkers()
    println_time_flush("removing workers $oldworkers")
    rmprocs(oldworkers)
    flush(stdout)
    if "SLURM_JOBID" in keys(ENV)
        num_cpus_to_request = parse(Int, ENV["SLURM_TASKS_PER_NODE"])
        println("requesting $(num_cpus_to_request) cpus from slurm.")
        flush(stdout)
        pids = addprocs_slurm(num_cpus_to_request)
    else
        pids = addprocs()
    end
    println_time_flush("Workers added: $pids")
    return pids
end

# ----------------------------------

@GenGlobal g_RemoteEstObj

# ----------------------------------

abstract type AbstractEstimationObjects end

struct RemoteEstObj{D<:DataSetofSets} <: AbstractEstimationObjects
    data::D
    llvec::SharedVector{Float64}
    scoremat::SharedMatrix{Float64}
    sim::SimulationDrawsMatrix{Float64,Matrix{Float64}}

    function RemoteEstObj(data::D,llvec,scoremat,sim) where {D}
        k, n = size(scoremat)
        n == num_i(data) == length(llvec) == num_i(sim) || throw(DimensionMismatch("num_i"))
        k == _nparm(data) || throw(DimensionMismatch("k"))
        _nparm(drill(data)) == _nparm(sim) || throw(DimensionMismatch("sim k"))
        return new{D}(data, llvec, scoremat, sim)
    end
end

struct LocalEstObj{D<:DataSetofSets} <: AbstractEstimationObjects
    data::D
    theta0::Vector{Float64}
    theta1::Vector{Float64}
    grad::Vector{Float64}
    hess::Matrix{Float64}
    invhess::Matrix{Float64}

    function LocalEstObj(data::D,theta0,theta1,grad,hess,invhess) where {D}
        k = length(theta0)
        k == length(theta1) == length(grad) || throw(DimensionMismatch())
        k == checksquare(hess) == checksquare(invhess) || throw(DimensionMismatch())
        k == _nparm(data) || throw(DimensionMismatch())

        return new{D}(data,theta0,theta1,grad,hess,invhess)
    end
end

data(x::AbstractEstimationObjects) = x.data

llvec(x::RemoteEstObj) = x.llvec
scoremat(x::RemoteEstObj) = x.scoremat
sim(x::RemoteEstObj) = x.sim
size(x::RemoteEstObj) = size(scoremat(x))
num_i(x::RemoteEstObj) = length(llvec(eo))
_nparm(x::RemoteEstObj) = size(scoremat(x),1)
LL(x::RemoteEstObj) = sum(llvec(x))
negLL(x::RemoteEstObj) = -LL(x)
loglikelihood(x::RemoteEstObj) = LL(x)


theta0(x::LocalEstObj) = x.theta0
theta1(x::LocalEstObj) = x.theta1
grad(  x::LocalEstObj) = x.grad
hess(  x::LocalEstObj) = x.hess
invhess(x::LocalEstObj) = x.invhess
num_i( x::LocalEstObj) = num_i(data(x))
_nparm(x::LocalEstObj) = length(theta0(x))

function reset!(x::LocalEstObj)
    fill!(grad(x), 0)
    fill!(hess(x), 0)
    fill!(invhess(x),0)
    fill!(theta1(x),0)
end


@noinline function update_reo!(reo::RemoteEstObj, theta::Vector)
    dat = data(reo)
    thetas = split_thetas(dat, theta)
    for (d, θ) in zip(dat, thetas)
        update!(d, θ)
    end
    return nothing
end

update_reo!(theta::Vector) = update_reo!(get_g_RemoteEstObj(), theta)

@noinline function reset_reo!(reo::RemoteEstObj)
    data_drill = drill(data(reo))
    ddm = _model(data_drill)
    vf = value_function(ddm)
    fill!(vf, 0)
end

reset_reo!() = reset_reo!(get_g_RemoteEstObj())


function invhess!(x::LocalEstObj)
    invhess(x) .= inv(hess(x))
    return invhess(x)
end

function RemoteEstObj(leo::LocalEstObj, M, pids = workers())
    n = num_i(leo)
    k = _nparm(leo)
    llvec    = SharedVector{Float64}(n;    pids=pids)
    scoremat = SharedMatrix{Float64}(k, n; pids=pids)
    sim      = SimulationDraws(M, data(leo))
    return RemoteEstObj(data(leo), llvec, scoremat, sim)
end

function LocalEstObj(data, theta)
    k = length(theta)
    k == _nparm(data) || throw(DimensionMismatch("length(theta) = $(length(theta)) but _nparm(data) = $(_nparm(data)) θ = $theta"))

    theta0 = Vector{Float64}(undef, k)
    theta1 = similar(theta0)
    grad   = similar(theta0)
    hess   = Matrix{Float64}(undef, k, k)
    invhess = similar(hess)
    return LocalEstObj(data, theta0, theta1, grad, hess, invhess)
end


struct EstimationWrapper{D} <: AbstractEstimationObjects
    leo::LocalEstObj{D}
    reo::RemoteEstObj{D}
    function EstimationWrapper(leo::LocalEstObj{D},reo::RemoteEstObj{D}) where {D}
        @assert data(leo) === data(reo)
        return new{D}(leo,reo)
    end
end

LocalEstObj(ew::EstimationWrapper) = ew.leo
RemoteEstObj(ew::EstimationWrapper) = ew.reo
data(ew::EstimationWrapper) = data(RemoteEstObj(ew))
OneTo(ew::EstimationWrapper) = OneTo(num_i(LocalEstObj(ew)))
_nparm(ew::EstimationWrapper) = _nparm(LocalEstObj(ew))

function check_theta(ew::EstimationWrapper, theta)
    length(theta) == _nparm(ew) || throw(DimensionMismatch())
end

function update!(ew::EstimationWrapper, theta, dograd)
    l = LocalEstObj(ew)
    r = RemoteEstObj(ew)
    theta1(l) .= theta
    if dograd
        score = scoremat(r)
        h = hess(l)
        g = grad(l)
        mul!(h, score, score')
        sum!(reshape(g, :, 1), score)
        # println("gradient = $g")
        g .*= -1
    end
    nll = negLL(r)
    countplus!(-nll,theta)
    flush(stdout)
    return nll
end


@noinline function simloglik!(i, theta, dograd, reo::RemoteEstObj; kwargs...)

    check_finite(theta)

    gradi = view(scoremat(reo), :, i)
    grptup = getindex.(data(reo), i)
    simi = view(sim(reo), i)
    θρ = theta_ρ(data(reo), theta)

    fill!(_qm(sim(reo)), 0)
    fill!(gradi, 0)

    # do updates
    @assert isfinite(θρ)
    update!(simi, θρ)

    thetasvw = split_thetas(data(reo), theta)
    idxs = theta_indexes(data(reo))

    llv = llvec(reo)
    llv[i] = simloglik!(gradi, grptup, thetasvw, idxs, simi, dograd; kwargs...)

end


simloglik!(i, theta, dograd; kwargs...) =
    simloglik!(i, theta, dograd, get_g_RemoteEstObj(); kwargs...)

function serial_simloglik!(ew, theta, dograd; kwargs...)
    check_theta(ew,theta)
    reo = RemoteEstObj(ew)
    update_reo!(reo, theta)
    map(i -> simloglik!(i, theta, dograd, reo; kwargs...), OneTo(ew))
    return update!(ew, theta, dograd)
end

function parallel_simloglik!(ew, theta, dograd; kwargs...)
    check_finite(theta)
    check_theta(ew,theta)
    wp = CachingPool(workers())
    @eval @everywhere update_reo!($theta)
    let x=theta, dograd=dograd, kwargs=kwargs
        pmap(i -> simloglik!(i, x, dograd; kwargs...), wp, OneTo(ew))
    end
    return update!(ew, theta, dograd)
end


function test_parallel_simloglik!(ew, theta, dograd; kwargs...)
    @everywhere reset_reo!()
    parallel_simloglik!(ew, theta, dograd; kwargs...)
end

function test_serial_simloglik!(ew, theta, dograd; kwargs...)
    reset_reo!(RemoteEstObj(ew))
    serial_simloglik!(ew, theta, dograd; kwargs...)
end
