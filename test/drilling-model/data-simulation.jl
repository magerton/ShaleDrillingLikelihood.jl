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
using Dates

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
    total_leases,
    zero_out_small_probs,
    lrvar,
    lrstd,
    lrmean,
    simulate

println("print to keep from blowing up")

@testset "Simulate Dynamic Drilling Model" begin

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    num_parms = _nparm(f)

    # dimensions of state space
    nψ, _dmax, nz =  13, 3, 21

    # number of obs
    num_zt = 100
    num_i = 5

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # price process
    # ----------------------------------

    # define price process
    zrho = 0.8 # AR1 parameter
    zprocess = AR1process(0.8, 1.33*(1-zrho), 0.265^2*(1-zrho^2))

    # simulate price process
    zvec = simulate(zprocess, num_zt)
    ztrng = range(Date(2003,10); step=Quarter(1), length=num_zt)
    _zchars = ExogTimeVars(tuple.(zvec), ztrng)

    # grid for price process
    zmin = min(minimum(zvec), lrmean(zprocess)-3*lrstd(zprocess))
    zmax = max(maximum(zvec), lrmean(zprocess)+3*lrstd(zprocess))
    zrng = range(zmin, zmax; length=nz)
    zs = ( zrng, )
    @test prod(length.(zs)) == nz

    # transition for price process
    ztrans = sparse(zero_out_small_probs(tauchen_1d(zprocess, zrng), 1e-4))
    # ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    # ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(_dmax,4,5,3,2)

    # ichars
    ichar = (4.0, 0.25,)

    # ddm object
    ddm_no_t1ev   = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, true)
    tmpv = DCDPTmpVars(ddm_no_t1ev)

    theta = [-4.0, -1.0, -3.0, 0.7]
    @test length(theta) == num_parms

    u,v = randn(num_i), randn(num_i)

    for ddm in (ddm_with_t1ev, ddm_with_t1ev,)
        datanew = ShaleDrillingLikelihood.DataDynamicDrill(
            u, v, _zchars, ddm, theta;
            minmaxleases=2:5,
            nper_initial=10:15,
            tstart=1:10
        )
    end

end

end # module
