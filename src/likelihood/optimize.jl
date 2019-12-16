using Optim: OnceDifferentiable

export solve_model


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
