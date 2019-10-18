function loglik_drill_lease(lease::DrillLease, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread, dograd::Bool=false)::T where {T}
    LL = zero(T)
    ubv = _ubv(dtv)
    grad = _grad(dtv)

    for obs in lease
        actions = actionspace(obs)
        n = length(actions)
        ubv_vw = view(ubv, OneTo(n))

        @inbounds @simd for d in actions
            ubv_vw[d+1] = full_payoff(d, obs, theta, sim)
        end

        if !dograd
            LL += ubv_vw[_y(obs)+1] - logsumexp(ubv_vw)
        else
            LL += ubv_vw[_y(obs)+1] - logsumexp_and_softmax!(ubv_vw)

            @inbounds for d in actions
                wt = d==_y(obs) ? 1-ubv_vw[d+1] : -ubv_vw[d+1]
                @inbounds @simd for k in eachindex(grad)
                    grad[k] += wt * dfull_payoff(k, d, obs, theta, sim)
                end
            end
        end
    end

    return LL
end

function loglik_drill_unit(unit::DrillUnit, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread, dograd::Bool=false)::T where {T}
    LL = zero(T)
    nJ = num_initial_leases(unit)
    grad = _grad(dtv)
    gradJ = view(_gradJ(dtv), :, nJ)
    LLj = view(_llj(dtv), OneTo(nJ))

    if nJ > 0
        for (ji,lease) in enumerate(InitialDrilling(unit))
            fill!(grad, 0)
            LLj[ji] = loglik_drill_lease(lease, theta, sim, dtv, dograd)
            if dograd
                gradJ[:,ji] .= grad   # FIXME
            end
        end
        if dograd
            LL += logsumexp_and_softmax!(LLj)
            mul!(grad, gradJ, LLj)
        else
            LL += logsumexp(LLj)
        end
    end

    for lease in DevelopmentDrilling(unit)
        LL += loglik_drill_lease(lease, theta, sim, dtv, dograd)
    end
    dograd && _drillgradm(dtv) .= _grad(dtv)
    return LL
end

function simloglik_drill!(grad::AbstractVector, unit::DrillUnit, theta::AbstractVector{T}, sims::SimulationDrawsVector, dtv::DrillingTmpVarsAll, dograd::Bool=false)::T where {T}
    M = _num_sim(sims)
    llm = _llm(sims)
    gradm = _drillgradm(sims)

    let M=M, llm=llm, unit=unit, theta=theta, sims=sims, dtv=dtv
        @threads for m in OneTo(M)
            local sims_m = sims[m]
            local dtvi = dtv[threadid()]
            llm[m] = loglik_drill_unit(unit, theta, sims_m, dtvi)
        end
    end

    if dograd
        LL = logsumexp_and_softmax!(llm) - log(M)
        grad .+= _drillgradm(sims) * llm
        return LL
    else
        return logsumexp(llm) - log(M)
    end
end



function logL!(grad::AbstractVector, data::DataDrill, sim::SimulationDrawsMatrix, dtv::DrillingTmpVarsAll, theta::AbstractVector, dograd::Bool=false)
    lik = 0.0
    update!(sim, theta_drill_ρ(_model(data), theta))
    for (i,unit) in enumerate(data)
        lik += simloglik_drill!(grad, unit, theta, view(sim, i), dtv, dograd)
    end
    return lik
end
