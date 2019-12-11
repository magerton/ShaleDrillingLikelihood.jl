module ShaleDrillingLikelihood_DynamicDrillingModelInterpolationTest

DOBTIME=false

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
    vw_extend,
    DataDynamicDrill,
    max_states,
    _D,
    simloglik!,
    simloglik_drill_data!,
    SimulationDraws,
    DCDPTmpVars,
    _model,
    theta_drill_ρ,
    update!,
    value_function,
    EV, dEV

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

    for ddm in (ddm_with_t1ev, ddm_no_t1ev,)
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

    discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

    # set up coefs
    θρ = 0.5
    αψ = 0.33
    αg = 0.56

    # set up coefs
    #            θρ  ψ        g    x_r     κ₁   κ₂
    θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
    #                αψ, αg, γx   σ2η, σ2u   # η is iwt, u is iw
    θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)
    #            drill  ext     α0  αg  αψ  θρ
    θ_drill_u = [-5.0, -0.85, -2.7, αg, αψ, θρ]
    θ_drill_c = vcat(θ_drill_u[1:3], θρ)

    # model
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=αg, α_ψ=αψ),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u
    @test _nparm(rwrd) == length(θ_drill_u)

    # parms
    num_i = 100

    # grid sizes
    nψ =  13
    _dmax = 3
    nz = 21

    # simulations
    M = 1_000

    # observations
    num_zt = 100
    obs_per_well = 10:20
    ddm_opts = (minmaxleases=2:5, nper_initial=40:50, tstart=1:20)

    # roylaty
    royalty_rates = [1/8, 3/16, 1/4,]
    num_royalty_rates = length(royalty_rates)
    nk_royalty = length(θ_royalty) - (length(royalty_rates)-1)-2

    # geology
    ogip_dist = Normal(4.68,0.31)

    # price parameters
    zrho = 0.8                # rho
    zmean = 1.33*(1-zrho)     # mu
    zvar = 0.265^2*(1-zrho^2) # sigsq

    # state space, psi-space
    wp = LeasedProblem(_dmax,4,5,3,2)
    ψs = range(-4.5; stop=4.5, length=nψ)

    # simulate price process
    zprocess = AR1process(zmean, zrho, zvar)
    zvec = simulate(zprocess, num_zt)
    ztrng = range(Date(2003,10); step=Quarter(1), length=num_zt)
    _zchars = ExogTimeVars(tuple.(zvec), ztrng)

    # price grid2
    zmin = min(minimum(zvec), lrmean(zprocess)-3*lrstd(zprocess))
    zmax = max(maximum(zvec), lrmean(zprocess)+3*lrstd(zprocess))
    zrng = range(zmin, zmax; length=nz)
    zs = tuple(zrng)
    @test prod(length.(zs)) == nz
    ztrans = sparse(zero_out_small_probs(tauchen_1d(zprocess, zrng), 1e-4))
    @test all(isfinite.(ztrans))

    # random vars
    log_ogip = rand(ogip_dist, num_i)
    Xroyalty = vcat(log_ogip', randn(nk_royalty-1, num_i))
    u,v = randn(num_i), randn(num_i)

    # model
    ddm_no_t1ev     = DynamicDrillingModel(rwrd_u, discount, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev   = DynamicDrillingModel(rwrd_u, discount, wp, zs, ztrans, ψs, true)
    ddm_c_no_t1ev   = DynamicDrillingModel(rwrd_c, discount, wp, zs, ztrans, ψs, false)
    ddm_c_with_t1ev = DynamicDrillingModel(rwrd_c, discount, wp, zs, ztrans, ψs, true)

    # tests about royalty coef
    @test theta_royalty_ρ(RoyaltyModel(), θ_royalty) == θρ

    # test drilling coef
    @test theta_ρ(rwrd, θ_drill_u) == θρ
    @test theta_ρ(revenue(rwrd), vw_revenue(rwrd, θ_drill_u)) == θρ
    @test θ_drill_u[1:1] == vw_cost(rwrd, θ_drill_u)
    @test θ_drill_u[2:2] == vw_extend(rwrd, θ_drill_u)
    @test θ_drill_u[3:end] == vw_revenue(rwrd, θ_drill_u)

    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)

    # construct ichars for drilling
    _ichars = [gr for gr in zip(log_ogip, royalty_rates[_y(data_roy)])]

    # construct drilling data
    data_drill_w = DataDynamicDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    data_drill_n = DataDynamicDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)

    # constrained versions of above
    data_drill_w_con = DataDrill(ddm_c_with_t1ev, data_drill_w)
    data_drill_n_con = DataDrill(ddm_c_no_t1ev, data_drill_n)

    # number of wells drilled
    nwells_w = map(s -> _D(wp,s), max_states(data_drill_w))
    nwells_n = map(s -> _D(wp,s), max_states(data_drill_n))

    @show sort(countmap(nwells_w))
    @show sort(countmap(nwells_n))

    data_produce_w = DataProduce(u, nwells_w, obs_per_well, θ_produce, log_ogip)
    data_produce_n = DataProduce(u, nwells_n, obs_per_well, θ_produce, log_ogip)

    @test sum(nwells_w .> 0) == length(Set(view(_x(data_produce_w), 1, :)))
    @test sum(nwells_n .> 0) == length(Set(view(_x(data_produce_n), 1, :)))

    @testset "Test dynamic Drilling model only" begin
        data = data_drill_w_con
        theta = θ_drill_c
        ddm = _model(data)
        vf = value_function(ddm)

        @test data isa DataDrill
        sim = SimulationDraws(1_000, data)
        nparm = _nparm(data)
        @test nparm == length(theta)
        grad = zeros(nparm)
        hess = zeros(nparm, nparm)
        tmpgrads = zeros(nparm, num_i)

        println("simloglik, no grad")
        simloglik_drill_data!(grad, hess, data, theta, sim, false)
        @test all(grad .== 0)
        println("simloglik, with grad")
        simloglik_drill_data!(grad, hess, data, theta, sim, true)
        @test all(isfinite.(EV(vf)))
        @test all(isfinite.(dEV(vf)))
        @test all(isfinite.(grad))

        println("simloglik finite diff")
        fd = Calculus.gradient(xx -> simloglik_drill_data!(grad, hess, data, xx, sim, false), theta, :central)

        fill!(grad,0)
        fill!(hess,0)
        println("simloglik gradient")
        simloglik_drill_data!(grad, hess, data, theta, sim, true, true)
        @test !all(grad.==0)
        @test all(isfinite.(grad))
        @test all(isfinite.(hess))
        @test isapprox(fd, grad; rtol=2e-5)
        println("done")

        if DOBTIME
            print("")
            @show @benchmark simloglik_drill_data!($grad, $hess, $data, $theta, $sim, false)
            @show @benchmark simloglik_drill_data!($grad, $hess, $data, $theta, $sim, true)
            print("")
        end


    end


end





end # module
