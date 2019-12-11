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
using Distributions
using Dates
using StatsBase

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
    simulate,
    theta_royalty_ρ,
    revenue,
    cost,
    extend,
    theta_ρ,
    vw_revenue,
    vw_cost,
    vw_extend

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
        _ichars = ichars_sample(ddm, num_i)
        datanew = ShaleDrillingLikelihood.DataDynamicDrill(
            u, v, _zchars, _ichars, ddm, theta;
            minmaxleases=2:5,
            nper_initial=10:15,
            tstart=1:10
        )
    end

end


@testset "simulate all data" begin

    Random.seed!(7)

    num_i = 100

    # set up coefs
    θρ = 0.5
    αψ = 0.33
    αg = 0.56

    # model
    rwrd = DrillReward(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())

    # parms
    nk_royalty = 2
    royalty_rates = [1/8, 3/16, 1/4,]
    num_royalty_rates = length(royalty_rates)

    # random vars
    log_ogip = rand(Normal(4.68,0.31), num_i)
    Xroyalty = vcat(log_ogip', randn(nk_royalty-1, num_i))
    u,v = randn(num_i), randn(num_i)

    # set up coefs
    #            θρ  ψ        g    x_r     κ₁   κ₂
    θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
    #                αψ, αg, γx   σ2η, σ2u   # η is iwt, u is iw
    θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)
    #            drill  ext     α0  αg  αψ  θρ
    θ_drill_u = [-6.0, -0.85, -2.7, αg, αψ, θρ]

    @test theta_royalty_ρ(RoyaltyModel(), θ_royalty) == θρ
    @test theta_ρ(rwrd, θ_drill_u) == θρ
    @test theta_ρ(revenue(rwrd), vw_revenue(rwrd, θ_drill_u)) == θρ
    @test θ_drill_u[1:1] == vw_cost(rwrd, θ_drill_u)
    @test θ_drill_u[2:2] == vw_extend(rwrd, θ_drill_u)
    @test θ_drill_u[3:end] == vw_revenue(rwrd, θ_drill_u)

    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)

    _ichars = [gr for gr in zip(log_ogip, royalty_rates[_y(data_roy)])]


end





end # module
