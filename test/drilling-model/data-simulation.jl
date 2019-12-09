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
    ssprime,
    total_leases

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

    # for balanced panel of states + decisions
    lease_states    = zeros(Int, num_zt, total_leases(data))
    lease_decisions = zeros(Int, num_zt, total_leases(data))
    lease_outcome   = Vector{Symbol}(undef, total_leases(data))

    # for start/end times of each lease
    tdrill          = zeros(Int,      total_leases(data))
    tend            = fill( num_zt,   total_leases(data))
    texplore_term   = fill( num_zt+1, total_leases(data))

    wp = statespace(ddm)
    terminal_states = (exploratory_terminal(wp), length(wp) )

    # create balance "panel"
    for unit in data
        for lease in InitialDrilling(unit)
            ztrng = ztrange(lease)
            j = _i(lease)
            lease_states[   ztrng, j] .= _x(lease)
            lease_decisions[ztrng, j] .= _y(lease)

            # if ever drilled
            if sum(_y(lease)) > 0
                obs_drilled_first = findfirst(ii -> ii > 0, _y(lease))
                tdrill[j] = ztrng[obs_drilled_first]
            end

            # if ever expired
            if exploratory_terminal(wp) in _x(lease)
                explore_terminal_obs = findfirst(ii -> ii == exploratory_terminal(wp), _x(lease))
                texplore_term[j] = ztrng[explore_terminal_obs]
            end

        end
    end

    selected_initial_leases = [j1_sample(unit) for unit in data]

    # now replace decisions with those from 1 particular lease in each unit
    for unit in data

        i = _i(unit) # unit number
        jselect = selected_initial_leases[i] # pick this lease

        y_jselect      = lease_decisions[:,jselect]  # pick these decisions
        x_jselect      = lease_states[:,jselect]     # pick this set of states
        tend_jselect   = tend[jselect]               # ending obs
        tdrill_jselect = tdrill[jselect]

        for lease in InitialDrilling(unit)
            j = _i(lease)
            date_j_leased = jtstart(data,j)

            drilled = 0 < tdrill_jselect
            drilled_b4_j_leased = tdrill_jselect < jtstart(data,j)
            expired_b4_j_drilled = texplore_term[j] <= tdrill_jselect

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

            # if unit is drilled and lease_j is expired at 1st drill, dorp
            if drilled
                exp_term = exploratory_terminal(wp)
                expire_b4_drill_jsel = lease_states[tdrill_jselect, j] == exp_term
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


tdrill_first    = fill(num_zt+1, total_leases(data))
tdrill_complete = fill(num_zt+1, total_leases(data))
texpire         = zeros(Int, total_leases(data))

# create balance "panel"
last_state = length(wp)
expired_state = exploratory_terminal(wp)

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

all_same_value(x) = all(x .== x[1])

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
    tstop1s = tstop1[joldrng]

    for (jidx, jnew) in enumerate(jnewrng)
        tnewrng = _tptr[jnew]:_tptr[jnew+1]-1
        toldrng = tstop1s[jidx]
        jolddidx = joldrng[jidx]
        xnew[tnewrng] .= lease_states[toldrng, jolddidx]
        ynew[tnewrng] .= lease_decisions[toldrng, jolddidx]
    end
end

ynew


unitids = zeros(Int, total_leases(data))
for i in 1:length(data)
    unitids[j1_range(data,i)] .= i
end

(unitids[keep_lease], ttimes1[keep_lease], ttimes2[keep_lease])
(unitids, vec(mapslices(sum, lease_decisions; dims=1)), ttimes1, ttimes2)



    jselect = selected_initial_leases[i] # pick this lease



    y_jselect      = lease_decisions[:,jselect]  # pick these decisions
    x_jselect      = lease_states[:,jselect]     # pick this set of states
    tend_jselect   = tend[jselect]               # ending obs
    tdrill_jselect = tdrill[jselect]



end



for j in 1:total_leases(data)
    if any(lease_decisions[:,j] .< 0 )
        all(lease_decisions[:,j] .== -1) || throw(error())
    end
end

lease_states[:,selected_initial_leases[2]]

hcat(lease_states[6:12,1:3], lease_decisions[6:12, 1:3])

exploratory_terminal(wp)

ShaleDrillingLikelihood.state(wp,11) # terminal

# lease_decisions[6:11, 1:3]
# lease_states[6:11, 1:3]
#
# tdrill[1:3]
#
# [ShaleDrillingLikelihood.state(wp,i) for i in 1:length(wp)]

# end # simulate ddm
#
#
#
# end # module
