function loglik_drill_lease(lease::DrillLease, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread)::T where {T}
    LL = zero(T)
    ubv = _ubv(dtv)
    for obs in lease
        actions = actionspace(obs)
        n = length(actions)
        @simd for d in actions
            ubv[d+1] = full_payoff(d, obs, theta, sim)
        end
        ubv_vw = view(ubv, OneTo(n))
        LL += ubv[_y(obs)+1] - logsumexp(ubv_vw)
    end
    return LL
end

function loglik_drill_unit(unit::DrillUnit, theta::AbstractVector{T}, sim::SimulationDraw, dtv::DrillingTmpVarsThread)::T where {T}
    LL = zero(T)
    nJ = num_initial_leases(unit)
    LLj = view(_llj(dtv), OneTo(nJ))

    if nJ > 0
        for (j,lease) in InitialDrilling(unit)
            LLj[j] = loglik_drill_lease(lease, theta, sim, dtv)
        end
        LL += logsumexp(LLj)
    end

    for lease in DevelopmentDrilling(unit)
        LL += loglik_drill_lease(lease, theta, sim, dtv)
    end

    return LL
end

function simloglik_drill!(unit::DrillUnit, theta::AbstractVector{T}, sims::SimulationDrawsVector, dtv::DrillingTmpVarsAll)::T where {T}
    M = length(sim)
    llm = _llm(sim)
    gradm = _drillgradm(sim)

    let M=M, llm=llm, unit=unit, theta=theta, sims=sims, dtv=dtv
        @threads for m in OneTo(M)
            local sim = sims[m]
            local dtvi = dtv[threadid()]
            llm[m] = loglik_drill_unit(unit, theta, sim, dtvi)
        end
    end

    return logsumexp(llm) - log(M)
end
