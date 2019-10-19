function gradweight(could_choose, did_choose, payoff)
    if did_choose == could_choose
        return 1-payoff
    else
        return -payoff
    end
end

"""
    loglik_drill_lease!(grad, lease, theta, sim, dtv, dograd)

Return log lik of `lease` history. Optionally *adds* to gradient vector `grad`.

Uses temp vector `ubv` from thread-specific `dtv::DrillingTmpVarsThread`
"""
function loglik_drill_lease!(
    grad::AbstractVector, lease::DrillLease,
    thet::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread{T},
    dograd::Bool
)::T where {T}


    LL = zero(T)
    ubv = _ubv(dtv)
    theta = _theta(dtv)

    for obs in lease
        actions = actionspace(obs)
        n = length(actions)
        ubv_vw = view(ubv, OneTo(n))

        @inbounds @simd for d in actions
            ubv[d+1] = full_payoff(d, obs, theta, sim)
        end

        tmp = ubv[_y(obs)+1]
        LL += tmp - logsumexp!(ubv_vw)

        if dograd
            @inbounds for d in actions
                wt = gradweight(d, _y(obs), ubv[d+1])
                @simd for k in eachindex(grad)
                    grad[k] += wt * dfull_payoff(k, d, obs, theta, sim)
                end # grad
            end # actions
        end # dograd

    end # obs

    return LL
end

"""
    loglik_drill_unit!(grad, unit, theta, sim, dtv, dograd)

Return log lik of `unit` history, integrating over `J`. possible leases.

If `dograd`, then `grad .+= ∇(log Lᵢ)` using thread-specific temp variables
`dtv::DrillingTmpVarsThread`. This overwrites `_gradJ(dtv)`
"""
function loglik_drill_unit!(
    grad::AbstractVector, unit::DrillUnit,
    theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread,
    dograd::Bool
)::T where {T}

    nJ    = num_initial_leases(unit)
    gradJ = view(_gradJ(dtv), :, OneTo(nJ))
    LLj   = view(_llj(dtv),      OneTo(nJ))

    fill!(gradJ, 0)

    LL = zero(T)
    if nJ > 0
        LLj .= log.(j1chars(InitialDrilling(unit)))
        for (ji,lease) in enumerate(InitialDrilling(unit))
            gradj = view(gradJ, :, ji)
            LLj[ji] += loglik_drill_lease!(gradj, lease, theta, sim, dtv, dograd)
        end
        LL += logsumexp!(LLj)
        dograd && BLAS.gemv!('N', 1.0, gradJ, LLj, 1.0, grad) # grad .+= gradJ * LLj
    end

    for lease in DevelopmentDrilling(unit)
        LL += loglik_drill_lease!(grad, lease, theta, sim, dtv, dograd)
    end

    return LL
end

"""
    simloglik_drill_unit!(grad, unit, theta, sim, dtv, dograd)

Compute *simulated* log Lᵢ of `unit` using `sims::SimulationDrawsVector`. If
`dograd`, update cols of `_drillgradm(sims) .+= ∇(log Lᵢₘ)`.

Uses set of thread-specific `dtv::DrillingTmpVarsAll`
"""
function simloglik_drill_unit!(
    unit::DrillUnit,
    theta::AbstractVector{T}, sims::SimulationDrawsVector,
    dtv::DrillingTmpVarsThread, dograd::Bool, dohess::Bool
)::T where {T}

    M = _num_sim(sims)
    llm   = _llm(dtv)
    gradM = _gradM(dtv)

    fill!(gradM, 0)

    let M=M, llm=llm, unit=unit, theta=theta, sims=sims, dtv=dtv, gradM=gradM
        # @threads
        for m in OneTo(M)
            # local dtvi = dtv[threadid()]  # thread-specific tmpvars
            local dtvi = dtv
            local sims_m = sims[m]        # get 1 particular simulation
            local gradm_i = view(gradM, :, m)
            llm[m] = loglik_drill_unit!(gradm_i, unit, theta, sims_m, dtvi, dograd)
        end
    end

    LL = logsumexp!(llm) - log(M)

    if dohess
        g = _gradtmp(dtv)
        mul!(g, gradM, llm)
        BLAS.ger!(1.0, g, g, _hess(dtv))
        _grad(dtv) .+= g
    elseif dograd
        BLAS.gemv!('N', 1.0, gradM, llm, 1.0, _grad(dtv)) # grad .+= gradM * llm
    end

    return LL
end



function simloglik_drill_data!(grad::AbstractVector, hess::AbstractMatrix, data::DataDrill,
    theta::AbstractVector{T}, sim::SimulationDrawsMatrix, dtv::DrillingTmpVarsAll,
    dograd::Bool=false, dohess::Bool=false
) where {T}

    dohess == true && dograd == false && throw(error("can't dohess without dograd"))
    checksquare(hess) == length(grad) == length(theta) || throw(DimensionMismatch("grad, theta incompatible"))

    reset!(dtv, theta)
    update!(sim, theta_drill_ρ(_model(data), theta))

    LL = Atomic{T}(zero(T))
    @threads for i in OneTo(length(data))
        uniti = data[i]
        dtvi = dtv[threadid()]
        simi = view(sim,i)
        LLi = simloglik_drill_unit!(uniti, theta, simi, dtvi, dograd, dohess)
        atomic_add!(LL, LLi)
    end

    if dograd
        fill!(grad, 0)
        dohess && fill!(hess, 0)
        for i in OneTo(nthreads())
            grad .+= _grad(dtv,i)
            dohess && (hess .+= _hess(dtv,i))
        end
    end

    return LL[]
end
