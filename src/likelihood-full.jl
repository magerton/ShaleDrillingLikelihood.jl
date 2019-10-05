using Base: OneTo

struct tmpvars where {V0<:AbstractVector}
    u::V0
    v::V0
    gradis::Matrix{T}
end

function simloglik!(grad, info_matrix, tmpvars, model, theta, data, dograd)

    check_theta_ok(model, theta, data) # check that royalty model is OK
    compute_xbetas!(tmpvars, data) # does x'β for royalty & production data

    LL = zero(eltype(theta))
    for indiv in data
        gradi = _gradi(tmpvars, i)
        LL += simloglik_i!(gradi, tmpvars, model, theta, indiv, dograd)
    end

    if dograd
        gradis = _gradis(tmpvars)
        sum!(reshape(gradis, :, 1), )
        mul!(info_matrix, gradis', gradis)
    end

    return LL
end


function simloglik_i!(grad::AbstractVector, tmpvars, model, theta::AbstractVector, data, dograd::Bool)

    dograd && zero_gradm!(tmpvars)

    # royalty rate
    rc = RoyaltyComputation
    simloglik_royalty!(rc, royalty(model), theta, dograd)

    # production
    plc = ProductionLikelihoodComputations(data)
    simloglik_produce!(LL, produce(model), theta, plc, _ψ2(tmpvars))

    # drilling
    solve_value_function(_vftmp(tmpvars), drilling(model), theta, itype(data), dograd)
    simloglik_threaded!(grad, tmpvars, model, thet, data, dograd)

    # log likelihood & gradient
    if !dograd
        LL = logsumexp(llm) - log(M)
    else
        LL = logsumexp_and_softmax!(llm) - log(M)

        # royalty rate
        update_grad_royalty!(grad, rc, royalty(model), theta)

        # likelihood
        pgc = ProductionGradientComputations
        update_grad_produce!(grad, produce(model), theta, plc, pgc)

        # drilling
        mul!(drilling_gradient, gradm, llm)
        grad .+= gradtmps[1]
    end

    isfinite(LL) || @warn "infinite simulated log likelihood" #  for unit $i, which had $(j1length(data,i)) leases. SLLi = $lli.  θ[rng_d] = $(θ[rng_d])"
    return LL
end
