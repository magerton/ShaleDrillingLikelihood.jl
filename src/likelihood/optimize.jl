using Optim: OnceDifferentiable

export RemoteEstObj,
    LocalEstObj,
    EstimationWrapper,
    solve_model,
    println_time_flush,
    update_reo!

function println_time_flush(str)
    println(Dates.format(now(), "HH:MM:SS   ") * str)
    flush(stdout)
end


@GenGlobal g_RemoteEstObj

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

theta0(x::LocalEstObj) = x.theta0
theta1(x::LocalEstObj) = x.theta1
grad(  x::LocalEstObj) = x.grad
hess(  x::LocalEstObj) = x.hess
invhess(x::LocalEstObj) = x.invhess
num_i( x::LocalEstObj) = num_i(data(x))
_nparm(x::LocalEstObj) = length(theta0(x))

function invhess!(x::LocalEstObj)
    invhess(leo) .= inv(hess(leo))
    return invhess(leo)
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
    k == _nparm(data) || throw(DimensionMismatch())

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
        g .*= -1
        h .*= -1
    end
    nll = negLL(r)
    countplus!(-nll,theta)
    return nll
end


@noinline function simloglik!(i, theta, dograd, reo::RemoteEstObj)

    gradi = view(scoremat(reo), :, i)
    grptup = getindex.(data(reo), i)
    simi = view(sim(reo), i)
    θρ = theta_ρ(data(reo), theta)

    fill!(_qm(sim(reo)), 0)
    fill!(gradi, 0)

    # do updates
    update!(simi, θρ)

    thetasvw = split_thetas(data(reo), theta)
    idxs = theta_indexes(data(reo))

    llv = llvec(reo)
    llv[i] = simloglik!(gradi, grptup, thetasvw, idxs, simi, dograd)

end

@noinline function update_reo!(reo::RemoteEstObj, theta::Vector)
    dat = data(reo)
    thetas = split_thetas(dat, theta)
    for (d, θ) in zip(dat, thetas)
        update!(d, θ)
    end
    return nothing
end

simloglik!(i, theta, dograd) = simloglik!(i,theta,dograd,get_g_RemoteEstObj())
update_reo!(theta::Vector) = update_reo!(get_g_RemoteEstObj(), theta)


function serial_simloglik!(ew, theta, dograd)
    check_theta(ew,theta)
    reo = RemoteEstObj(ew)
    update_reo!(reo, theta)
    map(i -> simloglik!(i, theta, dograd, reo), OneTo(ew))
    return update!(ew, theta, dograd)
end

function parallel_simloglik!(ew, theta, dograd)
    check_theta(ew,theta)
    wp = CachingPool(workers())
    @eval @everywhere update_reo!($theta)
    let theta=theta, dograd=dograd
        pmap(i -> simloglik!(i, theta, dograd), wp, OneTo(ew))
    end
    return update!(ew, theta, dograd)
end


function OnceDifferentiable(ew::EstimationWrapper, theta::Vector)

    function f(x::Vector)
        return parallel_simloglik!(ew, x, false)
        # return parallel_simloglik!(ew, x, false)
    end

    function fg!(g::Vector,x::Vector)
        nll = parallel_simloglik!(ew, x, true)
        # nll = serial_simloglik!(ew, x, true)
        g .= grad(LocalEstObj(ew))
        return nll
    end

    odfg = Optim.OnceDifferentiable(f, fg!, fg!, theta)
    return odfg
end

function invhessian!(ew::EstimationWrapper, theta)
    parallel_simloglik!(ew, theta, true)
    leo = LocalEstObj(ew)
    invhess(leo) .= inv(hess(leo))
    return invhess(leo)
end

function solve_model(ew, theta; allow_f_increases=true, show_trace=true, time_limit=60, kwargs...)
    leo = LocalEstObj(ew)
    theta0(leo) .= theta
    odfg  = OnceDifferentiable(ew, theta)
    resetcount!()

    # bfgs =  BFGS(;initial_invH = x -> invhessian!(ew, x))
    opts = Optim.Options(
        allow_f_increases=allow_f_increases,
        show_trace=show_trace,
        time_limit=time_limit,
        kwargs...
    )

    res = optimize(odfg, theta, BFGS(), opts)
    return res
end

export theta0, theta1, hess, invhess, grad, stderr, tstats, pvals, coef_and_se!

stderr!(leo) = sqrt.(diag(invhess!(leo)))
stderr(leo) = sqrt.(diag(invhess(leo)))
tstats!(leo) = theta1(leo) ./ stderr!(leo)
tstats(leo) = theta1(leo) ./ stderr(leo)
pvals!(leo) = cdf.(Normal(), -2*abs.(tstats!(leo)))
pvals(leo) = cdf.(Normal(), -2*abs.(tstats(leo)))

function Fstat!(leo; H0 = theta0(leo), alpha=0.05)
    k = _nparm(leo)
    err = theta1(leo) .- H0
    waldtest = err'*invhess!(leo)*err
    p = ccdf(Chisq(k, waldtest))
    reject = p < alpha
    return waldtest, p, reject
end

function coef_and_se!(leo)
    se = stderr!(leo)
    t = theta1(leo) ./ se
    p = pvals(leo)
    return hcat(theta1(leo), se, t, p)
end
