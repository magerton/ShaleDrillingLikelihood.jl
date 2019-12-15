using Optim: OnceDifferentiable

export RemoteEstObj,
    LocalEstObj,
    EstimationWrapper,
    solve_model,
    println_time_flush,
    update_reo!,
    strderr!, tstats!, pvals!,
    strderr, tstats, pvals,
    Fstat!

export theta0, theta1, hess, invhess, grad, stderr, tstats, pvals, coef_and_se!, coeftable

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
        g .*= -1
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
        reset!(LocalEstObj(ew))
        return parallel_simloglik!(ew, x, false)
        # return serial_simloglik!(ew, x, false)
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

function update_invhess!(leo::LocalEstObj)
    rnk = rank(hess(leo))
    k = checksquare(hess(leo))
    if rnk == k
        invhess(leo) .= inv(hess(leo))
    else
        @warn "rank of hess(leo) = $rnk < $k"
    end
    return invhess(leo)
end

function invhessian!(ew::EstimationWrapper, theta)
    parallel_simloglik!(ew, theta, true)
    leo = LocalEstObj(ew)
    return update_invhess!(leo)
end

const OptimOpts = Optim.Options(allow_f_increases=true, show_trace=true, time_limit=30)

# see https://github.com/JuliaNLSolvers/LineSearches.jl/tree/master/src
bfgs(ew) =  Optim.BFGS(;
    # linesearch = Optim.BackTracking(order=3),
    linesearch = Optim.MoreThuente(),
    initial_invH = x -> invhessian!(ew, x)
)
nelder() = Optim.NelderMead()

function solve_model(ew, theta; OptimOpts=OptimOpts, OptimMethod=bfgs(ew))
    leo = LocalEstObj(ew)
    theta0(leo) .= theta
    odfg  = OnceDifferentiable(ew, theta)
    resetcount!()
    res = optimize(odfg, theta, OptimMethod, OptimOpts)
    return res
end

coef(  leo::LocalEstObj) = theta1(leo)
stderr(leo::LocalEstObj) = sqrt.(diag(invhess(leo)))
tstats(leo::LocalEstObj) = coef(leo) ./ stderr(leo)

function pvals(leo::LocalEstObj)
    t = tstats(leo)
    p = 2 .* ccdf.(Normal(), abs.(t))
    return p
end

function coef_and_se(leo::LocalEstObj)
    se = stderr(leo)
    t = coef(leo) ./ se
    p = pvals(leo)
    return hcat(coef(leo), se, t, p)
end

tstats!(leo)      = (update_invhess!(leo); stderr(leo))
stderr!(leo)      = (update_invhess!(leo); tstats(leo))
pvals!(leo)       = (update_invhess!(leo); pvals(leo))
coef_and_se!(leo) = (update_invhess!(leo); coef_and_se(leo))

@deprecate coef_and_se(args...) coeftable(args...)
@deprecate coef_and_se!(args...) coeftable(args...)

function Fstat!(leo::LocalEstObj; H0 = theta0(leo), alpha=0.05)
    k = _nparm(leo)
    err = coef(leo) .- H0
    waldtest = err'*invhess!(leo)*err
    p = ccdf(Chisq(k), waldtest)
    reject = p < alpha
    return waldtest, p, reject
end

function critval(alpha::Real=0.05, twosided::Bool=true)
    0 < alpha < 1 || throw(DomainError(alpha))
    alpha >= 0.5 && @warn "α = $alpha > 0.05"
    a = twosided ? alpha/2 : alpha
    return quantile(Normal(), a)
end

function coeftable(leo, alpha::Real=0.05)
    # see defn at
    # https://github.com/JuliaStats/StatsBase.jl/blob/da42557642046116097ddaca39fd5dc2c41402cc/src/statmodels.jl#L376-L401
    update_invhess!(leo)
    cc = coef(leo)
    se = stderr(leo)
    tt = tstats(leo)
    p = pvals(leo)
    ci = se .* critval(alpha, true)

    omlev100 = (1 - alpha)*100
    levstr = isinteger(omlev100) ? string(Integer(omlev100)) : string(omlev100)

    tbl = hcat(cc, se, tt, p, cc+ci, cc-ci)
    cols = ["Estimate","Std. Error","t value","Pr(>|t|)","Lower $levstr%","Upper $levstr%"]
    rows = coefnames(data(leo))
    pvalcol = 4
    return CoefTable(tbl, cols, rows, pvalcol)
end
