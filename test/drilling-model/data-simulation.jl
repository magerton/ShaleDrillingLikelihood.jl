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
    j1_sample

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

    num_zt = 200

    data = DataDrill(ddm, theta;
        minmaxleases=2:5,
        num_i=5,
        nper_initial=10:15,
        nper_development=0:0,
        num_zt=num_zt,
        tstart=1:24,
        xdomain=0:0
    )

    lease_states    = zeros(Int, num_zt, length(j1chars(data)))
    lease_decisions = zeros(Int, num_zt, length(j1chars(data)))

    j = 0
    leaselen = zeros(Int,0)
    for unit in data
        for lease in InitialDrilling(unit)
            j+= 1
            push!(leaselen, length(lease))
            ztrng = ztrange(lease)
            @test length(ztrng) > 0
            @test _i(lease) == j
            lease_states[ztrng, _i(lease)] .= _x(lease)
            lease_decisions[ztrng, _i(lease)] .= _y(lease)
        end
    end
    @test j == length(j1chars(data))
    @test sum(leaselen) == length(data.x) == last(tptr(data))-1

    # @show lease_states[:,1]
    # @show lease_decisions[:,1]

    @show selected_initial_leases = collect(j1_sample(unit) for unit in data)
    @show data.j1ptr, data.j2ptr, data.tptr




end # simulate ddm



end # module
