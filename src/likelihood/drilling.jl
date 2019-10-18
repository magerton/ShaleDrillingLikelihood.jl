function loglik_drill_lease(lease::DrillLease, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread)::T where {T}
    LL = zero(T)
    ubv = _ubv(dtv)

    for obs in lease
        actions = actionspace(obs)
        n = length(actions)
        ubv_vw = view(ubv, OneTo(n))

        @inbounds @simd for d in actions
            ubv_vw[d+1] = full_payoff(d, obs, theta, sim)
        end

        LL += ubv_vw[_y(obs)+1] - logsumexp(ubv_vw)
    end

    return LL
end

function loglik_drill_unit(unit::DrillUnit, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread)::T where {T}
    LL = zero(T)
    nJ = num_initial_leases(unit)
    LLj = view(_llj(dtv), OneTo(nJ))

    if nJ > 0
        for (ji,lease) in enumerate(InitialDrilling(unit))
            LLj[ji] = loglik_drill_lease(lease, theta, sim, dtv)
        end
        LL += logsumexp(LLj)
    end

    for lease in DevelopmentDrilling(unit)
        LL += loglik_drill_lease(lease, theta, sim, dtv)
    end

    return LL
end

function simloglik_drill!(unit::DrillUnit, theta::AbstractVector{T}, sims::SimulationDrawsVector, dtv::DrillingTmpVarsAll)::T where {T}
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

    return logsumexp(llm) - log(M)
end



function logL(data::DataDrill, sim::SimulationDrawsMatrix, dtv::DrillingTmpVarsAll, theta::AbstractVector)
    lik = 0.0
    update!(sim, theta_drill_ρ(_model(data), theta))

    for (i,unit) in enumerate(data)
        lik += simloglik_drill!(unit, theta, view(sim, i), dtv)
    end
    return lik
end
