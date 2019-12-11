"""
    loglik_drill_lease!(grad, lease, theta, sim, dtv, dograd)

Return log lik of `lease` history. Optionally *adds* to gradient vector `grad`.

Uses temp vector `ubv` from thread-specific `dtv::DrillingTmpVarsThread`
"""
function loglik_drill_lease!(grad, lease, theta, sim, dtv, dograd)

    LL = azero(theta)
    ubv = _ubv(dtv)
    dubv = _dubv(dtv)

    for obs in lease
        actions = actionspace(obs)
        n = length(actions)
        resize!(ubv, n)
        dubv_vw = view(dubv, :, OneTo(n))

        @inbounds @simd for d in actions
            dp1 = d+1
            dubv_slice = view(dubv, :, dp1)
            ubv[dp1] = full_payoff!(dubv_slice, d, obs, theta, sim, dograd)
        end

        dp1 = _y(obs)+1
        LL += ubv[dp1] - logsumexp!(ubv)

        if dograd
            ubv[dp1] -= 1
            BLAS.gemv!('N', -1.0, dubv_vw, ubv, 1.0, grad)
        end
    end

    return LL
end

"""
    loglik_drill_unit!(grad, unit, theta, sim, dtv, dograd)

Return log lik of `unit` history, integrating over `J`. possible leases.

If `dograd`, then `grad .+= ∇(log Lᵢ)` using thread-specific temp variables
`dtv::DrillingTmpVarsThread`. This overwrites `_gradJ(dtv)`
"""
function loglik_drill_unit!(grad, unit, theta, sim, dtv, dograd)

    nJ    = num_initial_leases(unit)
    gradJ = view(_gradJ(dtv), :, OneTo(nJ))
    LLj = _llj(dtv)
    resize!(LLj, nJ)

    fill!(gradJ, 0)

    LL = azero(theta)
    if nJ > 0
        LLj .= log.(j1chars(InitialDrilling(unit)))
        for (ji,lease) in enumerate(InitialDrilling(unit))
            gradj = view(gradJ, :, ji)
            LLj[ji] += loglik_drill_lease!(gradj, lease, theta, sim, dtv, dograd)
        end
        LL += logsumexp!(LLj)

        dograd && BLAS.gemv!('N', 1.0, gradJ, LLj, 1.0, grad)

    end

    for lease in DevelopmentDrilling(unit)
        LL += loglik_drill_lease!(grad, lease, theta, sim, dtv, dograd)
    end

    return LL
end

"""
    simloglik!(grad, unit::DrillUnit, theta, sims, dtv, dograd)

Threaded computation of `_llm(sims)[m] += log lik(DrillUnit, sims[m])`
"""
function simloglik!(grad, unit::DrillUnit, theta, sims::SimulationDrawsVector, dograd)

    dtv = DrillingTmpVars(unit)
    M = _num_sim(sims)
    llm = _llm(sims)
    gradM = _drillgradm(sims)
    fill!(gradM, 0)

    length(grad) == length(theta) || throw(DimensionMismatch("grad,theta incompatible"))
    size(gradM, 1) == length(grad) || throw(DimensionMismatch("gradM not OK"))

    mapper = Mapper(M, 10)


    #FIXME: add update VF
    ddm = _model(_data(unit))
    tmpv = DCDPTmpVars(ddm)
    vf = value_function(ddm)
    fill!(EV(vf), 0)
    fill!(dEV(vf), 0)

    solve_vf_all!(tmpv, ddm, theta, ichars(unit), dograd)
    update_interpolation!(value_function(ddm), dograd)

    let M=M, llm=llm, unit=unit, theta=theta, sims=sims, dtv=dtv, gradM=gradM, mapper=mapper
        @threads for j in OneTo(nthreads())
            local dtvi = dtv[threadid()]  # thread-specific tmpvars
            while true
                mrng = nextrange!(mapper)
                isnothing(mrng) && break
                @inbounds for m in mrng
                    local sims_m = sims[m]        # get 1 particular simulation
                    local gradm_i = view(gradM, :, m)
                    llm[m] = loglik_drill_unit!(gradm_i, unit, theta, sims_m, dtvi, dograd)
                end
            end
        end
    end
end

function grad_simloglik!(grad, unit::DrillUnit, theta, sims::SimulationDrawsVector)
    M = _num_sim(sims)
    llm = _llm(sims)
    gradM = _drillgradm(sims)

    BLAS.gemv!('N', 1.0, gradM, llm, 1.0, grad)
end



"""
    simloglik_drill_unit!(grad, unit, theta, sim, dtv, dograd)

Compute *simulated* log Lᵢ of `unit` using `sims::SimulationDrawsVector`. If
`dograd`, update cols of `_drillgradm(sims) .+= ∇(log Lᵢₘ)`.

Uses set of thread-specific `dtv::DrillingTmpVarsAll`
"""
function simloglik_drill_unit!(grad, unit, theta, sims, dograd)
    simloglik!(grad, unit, theta, sims, dograd)
    LL = logsumexp!(_llm(sims)) - log(_num_sim(sims))
    dograd && grad_simloglik!(grad, unit, theta, sims)
    return LL
end


function simloglik_drill_data!(grad, hess, data, theta, sim::SimulationDrawsMatrix, dograd=false, dohess=false)

    dohess == true && dograd == false && throw(error("can't dohess without dograd"))
    checksquare(hess) == length(grad) == length(theta) || throw(DimensionMismatch("grad, theta incompatible"))

    update_theta!(DrillingTmpVars(data), theta)
    θρ = theta_drill_ρ(_model(data), theta)
    update!(sim, θρ)
    g = dohess ? similar(grad) : grad

    LL = azero(theta)
    for (i,unit) in enumerate(data)
        dohess && fill!(g, 0)
        LL += simloglik_drill_unit!(g, unit, theta, view(sim, i), dograd)
        if dohess
            BLAS.ger!(1.0, g, g, hess)
            grad .+= g
        end
    end

    return LL
end
