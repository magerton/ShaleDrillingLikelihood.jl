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

    theta = [-4.0, -1.0, -3.0, 0.7]
    @test length(theta) == num_parms

    num_zt = 100
    num_i = 5

    for ddm in (ddm_with_t1ev, ddm_with_t1ev,)
        datanew = ShaleDrillingLikelihood.DataDynamicDrill(
            randn(num_i),randn(num_i), ddm, theta;
            num_zt=num_zt,
            minmaxleases=2:5,
            nper_initial=10:15,
            tstart=1:10
        )
    end

end

end # module
