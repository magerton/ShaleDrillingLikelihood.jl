# module ShaleDrillingLikelihood_DynamicDrillingModelInterpolationTest


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
    ssprime

println("print to keep from blowing up")

# @testset "Simulate Dynamic Drilling Model" begin

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

    data = DataDrill(ddm, theta;
        minmaxleases=2:5,
        num_i=5,
        nper_initial=10:15,
        nper_development=0:0,
        num_zt=num_zt,
        tstart=1:10,
        xdomain=0:0
    )

    lease_states    = zeros(Int, num_zt, length(j1chars(data)))
    lease_decisions = zeros(Int, num_zt, length(j1chars(data)))
    tdrill          = zeros(Int,         length(j1chars(data)))
    tend            = fill( num_zt,      length(j1chars(data)))
    texplore_term   = fill(typemax(Int), length(j1chars(data)))

    wp = statespace(ddm)
    terminal_states = (exploratory_terminal(wp), length(wp) )

    # create balance "panel"
    for unit in data
        for lease in InitialDrilling(unit)
            ztrng = ztrange(lease)
            lease_states[   ztrng, _i(lease)] .= _x(lease)
            lease_decisions[ztrng, _i(lease)] .= _y(lease)

            if sum(_y(lease)) > 0
                obs_drilled_first = findfirst(ii -> ii > 0, _y(lease))
                tdrill[_i(lease)] = ztrng[obs_drilled_first]
            end

            if exploratory_terminal(wp) in _x(lease)
                explore_terminal_obs = findfirst(ii -> ii == exploratory_terminal(wp), _x(lease))
                texplore_term[_i(lease)] = ztrng[explore_terminal_obs]
            end

            if exploratory_terminal(wp) in _x(lease) || length(wp) in _x(lease)
                ending_obs = findfirst(ii -> ii ∈ terminal_states, _x(lease))
                tend[_i(lease)] = ztrng[ending_obs]
            end

        end
    end

    selected_initial_leases = collect(j1_sample(unit) for unit in data)

    lease_states[:,selected_initial_leases[2]]
    jtstart(data,6)
    tdrill[6]
    texplore_term[6]

    # now replace decisions with those from 1 particular lease in each unit
    for unit in data
        i = _i(unit) # unit number

        jselect = selected_initial_leases[i]    # pick this lease

        y_jselect = lease_decisions[:,jselect]  # pick these decisions
        x_jselect = lease_states[   :,jselect]  # pick this set of states
        tend_jselect = tend[jselect]            # ending obs
        tdrill_jselect = tdrill[jselect]

        for lease in InitialDrilling(unit)
            j = _i(lease)

            date_j_leased = jtstart(data,j)

            # if lease_j starts AFTER first drill, we drop
            if tdrill_jselect > 0 && jtstart(data,j) > tdrill_jselect
                lease_states[:,j] .= -1

            # if lease_j expires BEFORE we first drill
            elseif tdrill_jselect >  texplore_term[j]
                lease_states[:,j] .= -1

            else
                lease_decisions[:,j] .= y_jselect   # update other decisions
                for t in jtstart(data,j):size(lease_states,1)-1
                    st = lease_states[t,j]
                    d = lease_decisions[t,j]
                    lease_states[t+1,j] = ssprime(wp, st, d)
                end
            end
        end
    end

lease_states[:,selected_initial_leases[2]]

lease_states[:,1:3], lease_decisions[:, 1:3]

lease_decisions[6:11, 1:3]
lease_states[6:11, 1:3]

tdrill[1:3]

[ShaleDrillingLikelihood.state(wp,i) for i in 1:length(wp)]

# end # simulate ddm
#
#
#
# end # module