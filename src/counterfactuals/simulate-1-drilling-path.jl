function uview_col(x::SparseMatrixCSC, j::Integer)
    vals = nonzeros(x)
    rng = nzrange(x, j)
    return view(vals, rng)
end


"simulate a single path of drilling for section `i` given `uv` and `prob_uv`"
function simulate_lease!(simprim::SimulationPrimitives, lease::DrillLease, sim::SimulationDraw, weight::Number=1)

    simtmp = SimulationTmp(simprim)
    m = _model(DataDrill(lease))
    wp = statespace(m)
    θt = theta_drill(simprim)

    # reset sa and sb, which hold the current state
    # we start out in period 1 at state0
    reset!(simtmp, first(_x(lease)))

    # iterate over calendar periods from the lease signing to the
    #    end of the price vector
    for (obs,zt) in LeaseCounterfactual(lease)

        # Define temporary scalars
        @declarezero Float64 d0 d1 epsdeq1 epsdgt1 Prdeq1 Prdgt1
        @declarezero Float64 _profit surp cost rev extension

        eur, ext_cost = update!(simtmp, obs, sim, θt)

        tday, tmrw = today_tomorrow(simtmp, zt)
        _Eeps = dot(tday, Eeps(simtmp))

        # for si in _x(obs):length(wp) # OneTo(length(wp))
        for si in 1:length(wp) # OneTo(length(wp))
            PrTday = tday[si]

            if PrTday > 0
                action_probs = uview_col(Pprime(simtmp), si)
                actions = actionspace(wp, si)
                dmx = _dmax(wp,si)
                numactions = dmx+1

                # add to extension if we are expiring
                if si == end_ex1(wp)
                    extension += ext_cost * first(action_probs) * PrTday
                end

                # ϵ cost shocks so we can add them to observed drilling cost later
                if dmx >= 1
                    Pr1 = action_probs[2]
                    epsdeq1 += PrTday * xlogx(Pr1)
                    Prdeq1  += PrTday * Pr1
                    if dmx >= 2
                        Pr2 = view(action_probs, 3:numactions)
                        twodmx = 2:dmx
                        epsdgt1 += PrTday * sum(xlogx.(Pr2) ./ twodmx )
                        Prdgt1  += PrTday * sum(Pr2)
                    end
                end

                # expected drilling for D=0 vs D=1
                expected_d = dot(actions, action_probs) * PrTday
                if si <= end_ex0(wp) # before final expiration?
                    d0 += expected_d
                else
                    d1 += expected_d
                end

                # Expected payoffs
                dp1space_t = OneTo(numactions) # 0:dmax
                _profit += dot(action_probs, view(profit(simtmp),    dp1space_t)) * PrTday
                surp    += dot(action_probs, view(surplus(simtmp),   dp1space_t)) * PrTday
                cost    += dot(action_probs, view(drillcost(simtmp), dp1space_t)) * PrTday
                rev     += dot(action_probs, view(revenue(simtmp),   dp1space_t)) * PrTday
            end # PrTday > 0
        end # states


        # update our tracking vectors
        ds = SharedSimulations(simprim)

        update!(ds, zt, uniti(lease), weight,
            d0, d1, _ψ2(sim), eur,
            epsdeq1, epsdgt1, Prdeq1, Prdgt1,
            _Eeps, _profit, surp, rev, cost, extension
        )

        # Full transition
        @assert sum(tday) ≈ 1
        mul!(tmrw, Pprime(simtmp), tday)
        @assert sum(tmrw) ≈ 1

        # Distribution Pr(Dₜ = D|T) for D ∈ 1:Dmax after actions in T (eg sₜ₊₁)
        if zt == Tstop(simprim)-1
            i = uniti(lease)
            @inbounds for D in 1:_Dmax(wp)
                ds.D_at_T[D,i]  += weight * sum( view(tmrw, s_of_D(wp, D)) )
            end
        end

    end # loop over time
end # function


function update!(ds::SharedSimulations, t::Int, i::Int, weight,
        d0, d1, ψ2, eur,
        epsdeq1, epsdgt1, Prdeq1, Prdgt1,
        Eeps, profit, surplus, revenue, drillcost, extension
    )

    eursq = eur^2
    eurcub = eursq*eur

    # drilling, eur, ψ conditional on D==0
    d0 *= weight

    ds.d0[t,i]       += d0
    ds.d0psi[t,i]    += d0 * ψ2
    ds.d0eur[t,i]    += d0 * eur
    ds.d0eursq[t,i]  += d0 * eursq
    ds.d0eurcub[t,i] += d0 * eurcub

    # drilling, eur, ψ conditional on D>0
    d1 *= weight

    ds.d1[t,i]       += d1
    ds.d1psi[t,i]    += d1 * ψ2
    ds.d1eur[t,i]    += d1 * eur
    ds.d1eursq[t,i]  += d1 * eursq
    ds.d1eurcub[t,i] += d1 * eurcub

    # expected average cost shocks
    ds.epsdeq1[t,i]  += epsdeq1 * weight
    ds.epsdgt1[t,i]  += epsdgt1 * weight
    ds.Prdeq1[t,i]   += Prdeq1  * weight
    ds.Prdgt1[t,i]   += Prdgt1  * weight

    # profitability stuff
    ds.Eeps[t,i]      += Eeps      * weight
    ds.profit[t,i]    += profit    * weight
    ds.surplus[t,i]   += surplus   * weight
    ds.revenue[t,i]   += revenue   * weight
    ds.drillcost[t,i] += drillcost * weight
    ds.extension[t,i] += extension * weight


end
