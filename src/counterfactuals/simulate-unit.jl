@noinline function simulate_unit!(simprim::SimulationPrimitives, i, do_r=true)

    sim = SimulationDrawsVector(simprim, i)
    simtmp = SimulationTmp(simprim)
    theta_d, theta_r, theta_q = split_thetas(simprim)
    unit, obs_r, wells = getindex.(data(simprim), i)

    qm = _qm(sim)
    fill!(qm,0)

    M = _num_sim(sim)

    # royalty posterior
    if do_r
        simloglik_royalty!(first(obs_r), theta_r, sim, false)
        logsumexp!(qm)
    else
        invM = 1/M
        qm .= invM
    end

    model = _model(DataDrill(unit))
    solve_vf_and_update_itp!(model, theta_d, ichars(unit), false)
    @assert any( data(EVobj(value_function(model))) .!= 0 )

    # do update
    for m = OneTo(M)
        simm = getindex(sim, m)

        if num_initial_leases(unit) > 0
            for lease in InitialDrilling(unit)
                weight = qm[m] * j1chars(lease)
                simulate_lease!(simprim, lease, simm, weight)
            end
        else
            DD = DevelopmentDrilling(unit)
            @assert length(DD) == 1
            for lease in DD
                weight = qm[m]
                simulate_lease!(simprim, lease, simm, weight)
            end
        end

    end
end

function simulate_unit!(i, do_r=true)
     simprim = get_g_SimulationPrimitives()
     simulate_unit!(simprim, i, do_r)
end
