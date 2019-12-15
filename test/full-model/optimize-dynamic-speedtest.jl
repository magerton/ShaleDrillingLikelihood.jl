# module ShaleDrillingLikelihood_OptimizeDynamic

using Revise

DOBTIME = false
DOPROFILE = false

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using Test
using SparseArrays
using BenchmarkTools
using Calculus
using Random
using InteractiveUtils
using Distributions
using Dates
using StatsBase
using Optim
using CountPlus
using LinearAlgebra
using Profile
using Distributed
using CountPlus
# using PProf
using Juno
# if DOPROFILE
#     using ProfileView
# end

using Optim: minimizer, minimum, maximize, maximizer, maximum

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: ObservationDrill,
    SimulationDraw,
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
    simloglik!,
    SimulationDraws,
    _model,
    theta_drill_ρ,
    update!,
    value_function,
    EV, dEV,
    _x, _y,
    idx_produce_ψ, idx_drill_ψ,
    idx_produce_g, idx_drill_g,
    merge_thetas,
    split_thetas,
    theta_produce,
    serial_simloglik!,
    parallel_simloglik!,
    drill


# println("print to keep from blowing up")
#
#
# @testset "simulate all data" begin
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=AlphaG(), α_ψ=AlphaPsi()),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u

    θ_royalty = Theta(RoyaltyModel())
    θ_produce = Theta(ProductionModel())
    θ_drill_c = Theta(rwrd_c)
    θ_drill_u = Theta(rwrd_u)

    @test _nparm(rwrd) == length(θ_drill_u)

    using ShaleDrillingLikelihood.SDLParameters: TestPriceDates,
        TestPriceGrid, TestPriceTransition, TestAR1Process


    num_zt = 150          # drilling
    nz = 15

    tpd = TestPriceDates(num_zt)
    proc = TestAR1Process()
    simz = simulate(proc, num_zt)
    etv = ExogTimeVars(tuple.(simz), tpd)
    grid = TestPriceGrid(simz, nz; process=proc)
    trans = TestPriceTransition(grid...; process=proc)
    minimum(etv)
    length(product(grid))
    PriceProcess(proc, grid, etv, trans)

    supertype(supertype(AR1process))

    # parameters
    num_i = 50

    # grid sizes
    nψ =  13

    # simulations
    M = 50

    # observations
    obs_per_well = 10:20  # pdxn

    # roylaty
    royalty_rates = [1/8, 3/16, 1/4,]
    num_royalty_rates = length(royalty_rates)
    nk_royalty = length(θ_royalty) - (length(royalty_rates)-1)-2

    # geology
    ogip_dist = Normal(4.68, 0.31)

    # price parameters
    zrho = 0.8                # rho
    zmean = 1.33*(1-zrho)     # mu
    zvar = 0.265^2*(1-zrho^2) # sigsq

    # state space, psi-space
    # wp = LeasedProblem(_dmax,4,5,3,2)
    # wp = LeasedProblem(8, 8, 30, 20, 8)
    wp = LeasedProblem(8, 8, 12, 12, 8)
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


    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)
    royrates_endog = royalty_rates[_y(data_roy)]
    royrates_exog = sample(royalty_rates, num_i)

    # construct ichars for drilling
    _ichars = [gr for gr in zip(log_ogip, royrates_endog)]

    # construct drilling data
    # ddm_opts = (minmaxleases=1:1, nper_initial=40:40, tstart=1:50)
    # data_drill_w = DataDynamicDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    # data_drill_n = DataDynamicDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)

    # model
    ddm_no_t1ev     = DynamicDrillModel(rwrd_u, discount, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev   = DynamicDrillModel(rwrd_u, discount, wp, zs, ztrans, ψs, true)
    ddm_c_no_t1ev   = DynamicDrillModel(rwrd_c, discount, wp, zs, ztrans, ψs, false)
    ddm_c_with_t1ev = DynamicDrillModel(rwrd_c, discount, wp, zs, ztrans, ψs, true)

    ddm_opts = (minmaxleases=0:0, nper_initial=0:0, nper_development=60:60, tstart=1:75, xdomain=0:0)
    data_drill_w = DataDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    data_drill_n = DataDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)

    # constrained versions of above
    data_drill_w_con = DataDrill(ddm_c_with_t1ev, data_drill_w)
    data_drill_n_con = DataDrill(ddm_c_no_t1ev, data_drill_n)

    # number of wells drilled
    nwells_w = map(s -> _D(wp,s), max_states(data_drill_w))
    nwells_n = map(s -> _D(wp,s), max_states(data_drill_n))

    data_produce_w = DataProduce(u, nwells_w, obs_per_well, θ_produce, log_ogip)
    data_produce_n = DataProduce(u, nwells_n, obs_per_well, θ_produce, log_ogip)

    data_d_sm = data_drill_w_con
    data_d_lg = data_drill_w
    data_p    = data_produce_w
    data_r    = data_roy
    empty = EmptyDataSet()

    coef_links = [(idx_produce_ψ, idx_drill_ψ,), (idx_produce_g, idx_drill_g)]

    data_dril = DataSetofSets(data_d_sm, empty, empty)
    data_full = DataSetofSets(data_d_lg, data_r, data_p, coef_links)

    theta_dril = θ_drill_c
    theta_full = merge_thetas((θ_drill_u, θ_royalty, θ_produce), data_full)


    DOPAR = true

    if DOPAR
        rmprocs(workers())
        pids = addprocs()
    else
        pids = [1,]
    end
    println_time_flush("Putting pkg on workers")
    @everywhere using ShaleDrillingLikelihood
    println_time_flush("Package on workers")

    dataversions  = [data_dril, data_full]
    thetaversions = [theta_dril, theta_full]
    maxtimes = [10, 10] .* 60
