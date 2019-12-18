export  println_time_flush,
    strderr!, tstats!, pvals!,
    strderr, tstats, pvals,
    Fstat!,
    _model,
    coeftable


function println_time_flush(str)
    println(Dates.format(now(), "HH:MM:SS   ") * str)
    flush(stdout)
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


function solve_model(d::DataSetofSets, theta, M, maxtime)
    leo = LocalEstObj(d, theta)
    reo = RemoteEstObj(leo, M)
    ew = EstimationWrapper(leo, reo)
    leograd = grad(leo)
    @eval @everywhere set_g_RemoteEstObj($reo)

    resetcount!()
    startcount!([100, 500, 100000,], [1, 5, 100,])
    opts = Optim.Options(show_trace=true, time_limit=maxtime, allow_f_increases=true)

    res = solve_model(ew, theta; OptimOpts=opts)

    # println(res)
    println("Recomputing final gradient / hessian")
    let dograd=true, theta=Optim.minimizer(res)
        parallel_simloglik!(ew, theta, dograd)
        update!(ew, theta, dograd)
    end
    println(coeftable(leo))
    print("Parameter estimates are\n\t")
    print(sprintf_binary(Optim.minimizer(res)))
    print("\n")
    return res, ew
end
