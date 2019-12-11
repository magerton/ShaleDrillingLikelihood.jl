all_same_value(x) = all(x .== first(x))
range_i_to_ip1(x,i) = x[i] : (x[i+1]-1)

function DataDynamicDrill(u,v,_zchars::ExogTimeVars,_ichars::Vector{<:Tuple}, ddm::DynamicDrillingModel,theta;
        # num_zt=30,
        minmaxleases=0:3, nper_initial=50:50,
        tstart=5:15
    )

    num_i = length(u)
    @assert length(v) == num_i
    num_zt = length(_zchars)

    # _zchars = ExogTimeVarsSample(ddm, num_zt)

    data = DataDrill(u, v, _zchars, _ichars, ddm, theta;
        minmaxleases=minmaxleases,
        nper_initial=nper_initial,
        nper_development=0:0,
        tstart=tstart,
        xdomain=0:0
    )

    @assert num_i == length(data)

    # for balanced panel of states + decisions
    lease_states    = zeros(Int, num_zt, total_leases(data))
    lease_decisions = zeros(Int, num_zt, total_leases(data))

    # for start/end times of each lease
    tdrill  = zeros(Int,      total_leases(data)) # when lease j is 1st drilled
    texpire = fill( num_zt+1, total_leases(data)) # when lease j did expire

    # for start/end times of constructed data
    tstop1 = zeros(Int, total_leases(data)) # end of exploratory period
    tstop2 = zeros(Int, total_leases(data)) # end of infill period

    # state space
    wp = statespace(ddm)
    last_state = length(wp)
    expired_state = exploratory_terminal(wp)

    # choose one lease per unit
    selected_initial_leases = [j1_sample(unit) for unit in data]

    # create balanced "panel"
    for unit in data
        for lease in InitialDrilling(unit)

            ztrng = ztrange(lease)
            j = _i(lease)
            lease_states[   ztrng, j] .= _x(lease)
            lease_decisions[ztrng, j] .= _y(lease)

            # if ever drilled
            if lease_ever_drilled(lease)
                obs_drill_first = findfirst(d -> d > 0, _y(lease))
                tdrill[j] = ztrng[obs_drill_first]
            end

            # if ever expired
            if lease_expired(lease)
                obs_expire = findfirst(s -> s == exploratory_terminal(wp), _x(lease))
                texpire[j] = ztrng[obs_expire]
            end

        end
    end

    # now replace decisions with those from 1 particular lease in each unit
    for unit in data

        jselect = selected_initial_leases[_i(unit)] # pick this lease

        y_jselect      = lease_decisions[:,jselect]  # pick these decisions
        x_jselect      = lease_states[:,jselect]     # pick this set of states
        tdrill_jselect = tdrill[jselect]             # 1st drilling

        for lease in InitialDrilling(unit)
            j = _i(lease)

            decisions = view(lease_decisions, :, j)
            states = view(lease_states, :, j)

            # Boolean vars to test for events
            drilled  = 0 < tdrill_jselect
            drilled_b4_j_leased = tdrill_jselect < jtstart(data,j)
            expired_b4_j_drilled = texpire[j] <= tdrill_jselect

            #------------------------------
            # update lease_j with selected lease
            #------------------------------

            # if lease_j starts AFTER first drill, we drop
            if drilled && drilled_b4_j_leased
                decisions .= states .= -1

            # if lease_j expires BEFORE we first drill
            elseif expired_b4_j_drilled
                decisions .= states .= -1

            # update lease_j
            else
                decisions .= y_jselect   # update other decisions
                tdrill[j] = tdrill_jselect
                for t in jtstart(data,j):num_zt-1
                    states[t+1] = ssprime(wp, states[t], decisions[t])
                end
            end

            # unit drilled but lease_j has expired at 1st drill. then drop
            if drilled
                if lease_states[tdrill_jselect, j] == expired_state
                    decisions .= states .= -1
                end
            end

            #------------------------------
            # new start/end times for lease_j
            #------------------------------

            # if ever drilled
            if sum(decisions) > 0
                tstop1[j] = findfirst(d -> d > 0, decisions)

                # totally drilled
                if last_state in states
                    tstop2[j] = findfirst(s -> s == last_state, states)-1

                # not totally drilled
                else
                    tstop2[j] = length(states)
                end

            # else never drilled
            else
                # lease expires
                if expired_state in states
                    tstop1[j] = findfirst(s -> s == expired_state, states)
                # or not
                else
                    tstop1[j] = length(states)
                end
                tstop2[j] = tstop1[j]
            end
        end
    end

    # exploratory & infill drilling periods for each lease_j
    ttimes1 = [a:b for (a,b) in zip(jtstart(data)[1:total_leases(data)], tstop1)]
    ttimes2 = [a+1:b for (a,b) in zip(tstop1, tstop2)]

    # number of wells drilled for each lease_j
    num_wells  = vec(mapslices(sum, lease_decisions; dims=1))

    # do we keep this lease?
    keep_lease = vec(mapslices( x -> all(x.>=0), lease_decisions; dims=1))
    all(keep_lease[selected_initial_leases]) || throw(error("not keeping selected leases. boo!"))

    #------------------------------
    # checks that data are correct
    #------------------------------
    for unit in data
        i = _i(unit)
        jrng = j1_range(data, i)
        tstop1s = tstop1[jrng]
        keeps = keep_lease[jrng]
        drilled = num_wells[jrng] .> 0

        # for leases we have drilled & keep, exploratory periods end at same time
        if any(drilled[keeps])
            @assert all_same_value(tstop1s[keeps])

        # otherwise we don't have infill periods
        else
            @assert all(length.( ttimes2[jrng][keeps] ) .== 0)
        end
    end

    #------------------------------
    # create new dataset
    #------------------------------

    # @show ttimes2[selected_initial_leases]
    _tptr  = 1 .+ cumsum(vcat(0, length.(ttimes1[keep_lease]), length.(ttimes2[selected_initial_leases])))
    _j1ptr = 1 .+ cumsum(vcat(0, [sum(keep_lease[j1_range(data,i)]) for i in 1:num_i]))
    _j2ptr = (last(_j1ptr)-1) .+ (1:num_i)
    _jtstart = vcat(first.(ttimes1[keep_lease]), first.(ttimes2[selected_initial_leases]))
    _j1chars = j1chars(data)[keep_lease]

    _jtend = vcat(last.(ttimes1[keep_lease]), last.(ttimes2[selected_initial_leases]))
    @assert all(_jtend .<= num_zt)

    nobs = last(_tptr)-1
    xnew = zeros(Int, nobs)
    ynew = zeros(Int, nobs)

    for i = 1:num_i
        jnewrng = range_i_to_ip1(_j1ptr, i)

        # ensure these sum to 1
        _j1chars[jnewrng] ./= sum(_j1chars[jnewrng])

        keeps = keep_lease[j1_range(data,i)]
        joldrng = collect(j1_range(data,i))[keeps]
        ttimes1_jold = ttimes1[joldrng]

        selected_j = selected_initial_leases[i]

        for (jidx, jnew) in enumerate(jnewrng)
            tnewrng = range_i_to_ip1(_tptr, jnew)
            toldrng = ttimes1_jold[jidx]
            jolddidx = joldrng[jidx]
            length(tnewrng) == length(toldrng) || throw(error("i=$i, jnew=$jnew, jidx=$jidx, tnew=$tnewrng vs $toldrng"))
            xnew[tnewrng] .= lease_states[   toldrng, jolddidx]
            ynew[tnewrng] .= lease_decisions[toldrng, jolddidx]
        end

        let tnewrng = range_i_to_ip1(_tptr, _j2ptr[i])
            tinfill_i = ttimes2[selected_j]
            @assert length(tinfill_i) == length(tnewrng)
            xnew[tnewrng] .= lease_states[   tinfill_i, selected_j]
            ynew[tnewrng] .= lease_decisions[tinfill_i, selected_j]
        end
    end

    @assert num_i ≈ sum(_j1chars)

    new_data = DataDrill(
        ddm, _j1ptr, _j2ptr, _tptr, _jtstart,
        _j1chars, ichars(data), ynew, xnew, zchars(data)
    )

    return new_data
end
