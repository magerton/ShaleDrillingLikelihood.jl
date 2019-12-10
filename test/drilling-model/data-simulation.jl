module ShaleDrillingLikelihood_DynamicDrillingModelInterpolationTest


using ShaleDrillingLikelihood
using Test
using SparseArrays
using BenchmarkTools
using Calculus
using Random
using Profile
# using ProfileView
using InteractiveUtils

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: ObservationDrill,
    SimulationDraw,
    idx_ρ,
    actionspace,
    ichars_sample,
    j1chars,
    tptr,
    _x, _y,
    zchars,
    ztrange,
    InitialDrilling,
    _i,
    j1_sample,
    j1_range,
    jtstart,
    exploratory_terminal,
    ssprime,
    total_leases

println("print to keep from blowing up")

@testset "Simulate Dynamic Drilling Model" begin

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    num_parms = _nparm(f)

    # @show theta

    nψ, dmx, nz =  13, 3, 5

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-4.5; stop=4.5, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (4.0, 0.25,)

    # ddm object
    ddm_no_t1ev   = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, true)
    tmpv = DCDPTmpVars(ddm_no_t1ev)

    ddm = ddm_no_t1ev

    theta = [-4.0, -1.0, -3.0, 0.7]
    @test length(theta) == num_parms

    num_zt = 30
    num_i = 5

    data = DataDrill(ddm, theta;
        minmaxleases=2:5,
        num_i=num_i,
        nper_initial=10:15,
        nper_development=0:0,
        num_zt=num_zt,
        tstart=1:10,
        xdomain=0:0
    )

    # for balanced panel of states + decisions
    lease_states    = zeros(Int, num_zt, total_leases(data))
    lease_decisions = zeros(Int, num_zt, total_leases(data))

    # for start/end times of each lease
    tdrill  = zeros(Int,      total_leases(data))
    tend    = fill( num_zt,   total_leases(data))
    texpire = fill( num_zt+1, total_leases(data))

    # state space
    wp = statespace(ddm)
    last_state = length(wp)
    expired_state = exploratory_terminal(wp)

    # one lease per unit
    selected_initial_leases = [j1_sample(unit) for unit in data]

    all_same_value(x) = all(x .== x[1])

    # create balanced "panel"
    for unit in data
        for lease in InitialDrilling(unit)
            ztrng = ztrange(lease)
            j = _i(lease)
            lease_states[   ztrng, j] .= _x(lease)
            lease_decisions[ztrng, j] .= _y(lease)

            # if ever drilled
            if sum(_y(lease)) > 0
                obs_drill_first = findfirst(d -> d > 0, _y(lease))
                tdrill[j] = ztrng[obs_drill_first]
            end

            # if ever expired
            if exploratory_terminal(wp) in _x(lease)
                obs_expire = findfirst(s -> s == exploratory_terminal(wp), _x(lease))
                texpire[j] = ztrng[obs_expire]
            end

        end
    end

    # now replace decisions with those from 1 particular lease in each unit
    for unit in data

        i = _i(unit) # unit number
        jselect = selected_initial_leases[i] # pick this lease

        y_jselect      = lease_decisions[:,jselect]  # pick these decisions
        x_jselect      = lease_states[:,jselect]     # pick this set of states
        tend_jselect   = tend[jselect]               # ending obs
        tdrill_jselect = tdrill[jselect]             # 1st drilling

        for lease in InitialDrilling(unit)
            j = _i(lease)

            # Boolean vars to test for events
            drilled  = 0 < tdrill_jselect
            drilled_b4_j_leased = tdrill_jselect < jtstart(data,j)
            expired_b4_j_drilled = texpire[j] <= tdrill_jselect

            # if lease_j starts AFTER first drill, we drop
            if drilled && drilled_b4_j_leased
                lease_states[:,j] .= -1
                lease_decisions[:,j] .= -1

            # if lease_j expires BEFORE we first drill
            elseif expired_b4_j_drilled
                lease_states[:,j] .= -1
                lease_decisions[:,j] .= -1

            # update lease_j
            else
                lease_decisions[:,j] .= y_jselect   # update other decisions
                tdrill[j] = tdrill_jselect
                for t in jtstart(data,j):size(lease_states,1)-1
                    st = lease_states[t,j]
                    d = lease_decisions[t,j]
                    lease_states[t+1,j] = ssprime(wp, st, d)
                end
            end

            # unit drilled but lease_j has expired at 1st drill. then drop
            if tdrill_jselect > 0
                expire_b4_drill_jsel = lease_states[tdrill_jselect, j] == expired_state
                if expire_b4_drill_jsel
                    lease_states[:,j] .= -1
                    lease_decisions[:,j] .= -1
                end
            end

        end
    end

    # remake sample...
    # 1. drop invalid leases
    # 2. fix probabilitities in j1chars
    # 3. push lease_states[jtstart(data,j):tdrill] to tchars
    #      also update tptr, j1ptr
    # 4. for selected lease, push to tptr, tchars

    @test vec(mapslices( x -> all(x.==-1), lease_decisions; dims=1)) == vec(mapslices( x -> any(x.<0), lease_decisions; dims=1))
    @test vec(mapslices( x -> all(x.==-1), lease_states; dims=1)) == vec(mapslices( x -> any(x.<0), lease_states; dims=1))
    @test vec(mapslices( x -> all(x.==-1), lease_states; dims=1)) == vec(mapslices( x -> any(x.<0), lease_decisions; dims=1))


    keep_lease = vec(mapslices( x -> all(x.>=0), lease_decisions; dims=1))
    num_wells  = vec(mapslices(sum, lease_decisions; dims=1))
    tstop1 = zeros(Int, total_leases(data))
    tstop2 = zeros(Int, total_leases(data))
    all(keep_lease[selected_initial_leases]) || throw(error("not keeping selected leases. boo!"))


    # Find stop times for each lease / regime
    for unit in data
        for lease in InitialDrilling(unit)
            j = _i(lease)
            decisions = view(lease_decisions, :, j)
            states = view(lease_states, :, j)

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

    ttimes1 = [a:b for (a,b) in zip(jtstart(data)[1:total_leases(data)], tstop1)]
    ttimes2 = [a+1:b for (a,b) in zip(tstop1, tstop2)]


    for unit in data
        i = _i(unit) # unit number
        jrng = j1_range(data, i)

        tstop1s = tstop1[jrng]
        keeps = keep_lease[jrng]
        drilled = num_wells[jrng] .> 0
        if any(drilled[keeps])
            @test all_same_value(tstop1s[keeps])
        else
            @test all(length.( ttimes2[jrng][keeps] ) .== 0)
        end
    end


    num_i = length(data)
    _tptr  = 1 .+ cumsum(vcat(0, length.(ttimes1[keep_lease]), length.(ttimes2[selected_initial_leases])))
    _j1ptr = 1 .+ cumsum(vcat(0, [sum(keep_lease[j1_range(data,i)]) for i in 1:num_i]))
    _j2ptr = (last(_j1ptr)-1) .+ (1:num_i)
    nobs = last(_tptr)-1
    _jtstart = vcat(first.(ttimes1[keep_lease]), first.(ttimes2[selected_initial_leases]))

    xnew = zeros(Int, nobs)
    ynew = zeros(Int, nobs)

    for i = 1:num_i
        jnewrng = _j1ptr[i]:_j1ptr[i+1]-1

        keeps = keep_lease[j1_range(data,i)]
        joldrng = collect(j1_range(data,i))[keeps]
        ttimes1_jold = ttimes1[joldrng]

        for (jidx, jnew) in enumerate(jnewrng)
            tnewrng = _tptr[jnew]:_tptr[jnew+1]-1
            toldrng = ttimes1_jold[jidx]
            length(tnewrng) == length(toldrng) || throw(error("i=$i, jnew=$jnew, jidx=$jidx, tnew=$tnewrng vs $toldrng"))
            jolddidx = joldrng[jidx]
            xnew[tnewrng] .= lease_states[toldrng, jolddidx]
            ynew[tnewrng] .= lease_decisions[toldrng, jolddidx]
        end

        let tnewrng = _tptr[_j2ptr[i]]:_tptr[_j2ptr[i]+1]-1
            tinfill_i = ttimes2[selected_initial_leases[i]]
            xnew[tnewrng] = lease_states[   tinfill_i, selected_initial_leases[i]]
            ynew[tnewrng] = lease_decisions[tinfill_i, selected_initial_leases[i]]
        end
    end


    datanew = ShaleDrillingLikelihood.DataDynamicDrill(
        randn(num_i),randn(num_i), ddm, theta;
        num_zt=num_zt,
        minmaxleases=2:5,
        nper_initial=10:15,
        tstart=1:10
    )

end

end # module
