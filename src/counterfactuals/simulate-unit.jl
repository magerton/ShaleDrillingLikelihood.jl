@noinline function simulate_unit!(simprim::SimulationPrimitives, i, do_r=true)

    sim = SimulationDrawsVector(simprim, i)
    simtmp = SimulationTmp(simprim)
    theta_d, theta_r, theta_q = split_thetas(simprim)
    unit, obs_r, wells = getindex.(data(simprim), i)

    qm = _qm(sim)

    # royalty posterior
    if do_r
        simloglik_royalty!(first(obs_r), theta_r, sim, false)
        logsumexp!(qm)
    else
        invM = 1/_num_sim(sim)
        qm .= invM
    end

    m = _model(DataDrill(unit))
    solve_vf_and_update_itp!(m, theta_d, ichars(unit), false)

    # do update
    for m = OneTo(_num_sim(sim))
        simm = getindex(sim, m)

        if num_initial_leases(unit) > 0
            for lease in InitialDrilling(unit)
                weight = qm[m] * j1chars(lease)
                simulate_lease!(simprim, lease, simm, weight)
            end
        else
            for lease in DevelopmentDrilling(unit)
                weight = qm[m]
                simulate_lease!(simprim, lease, simm, weight)
            end
        end

    end
end

function simulate_m_drilling_paths!(i, do_r=true)
     simprim = g_SimulationPrimitives()
     simulate_m_drilling_paths!(simprim, i, do_r)
end
