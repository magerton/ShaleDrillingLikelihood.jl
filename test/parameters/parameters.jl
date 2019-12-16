module ShaleDrillingLikelihood_Parameters

using Revise

DOBTIME = false
DOPROFILE = false

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using Test
using SparseArrays
using Random
using Distributions
using Dates
using StatsBase
using LinearAlgebra

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
    drill,
    total_wells_drilled,
    _D

using ShaleDrillingLikelihood.SDLParameters: TestPriceDates,
    TestPriceGrid, TestPriceTransition, TestAR1Process,
    grid, series, transition

println("print to keep from blowing up")


@testset "Parameters" begin
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=AlphaG(), α_ψ=AlphaPsi()),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u

    θ_royalty = Theta(RoyaltyModel())
    θ_produce = Theta(ProductionModel())
    θ_drill_c = Theta(rwrd_c)
    θ_drill_u = Theta(rwrd_u)

    @test _nparm(rwrd) == length(θ_drill_u)


    num_zt = 150          # drilling
    nz = 15
    num_i = 50
    nψ =  13
    M = 50

    # prices
    tpd = TestPriceDates(num_zt)
    proc = TestAR1Process()
    simz = simulate(proc, num_zt)
    etv = ExogTimeVars(tuple.(simz), tpd)
    grd = TestPriceGrid(simz, nz; process=proc)
    trans = TestPriceTransition(grd...; process=proc)
    PriceProcess(proc, grd, etv, trans)

    @test minimum(etv) isa Tuple
    @test length(product(grd...)) == nz

    price_process = TestPriceProcess(;process=proc, nsim=num_zt, ngrid=nz)
    _zchars = SDLParameters.zchars(price_process)

    # ogips
    ogip_dist = NormalOGIP()
    @test ogip_dist == Normal(4.68, 0.31)
    logOGIP(num_i, Normal())
    log_ogip = logOGIP(num_i)

    # observations
    obs_per_well = 10:20  # pdxn

    # roylaty
    royalty_rates = [1/8, 3/16, 1/4,]
    @test royalty_rates == RoyaltyRates(3)
    @test royalty_rates == RoyaltyRates(:small)
    @test RoyaltyRates(6) == RoyaltyRates(:full)
    @test length(RoyaltyRates(6)) == 6
    @test issorted(RoyaltyRates(6))
    @test issorted(RoyaltyRates(3))

    Theta(RoyaltyModel())

    RoyaltyExogVars(num_i, royalty_rates, θ_royalty)
    RoyaltyExogVars(num_i, royalty_rates, θ_royalty, log_ogip)
    Xroyalty = RoyaltyExogVars(num_i, royalty_rates, θ_royalty, log_ogip, RoyaltyModel())


    # geology
    ψs = PsiSpace(nψ)
    @test ψs == range(-4.5; stop=4.5, length=nψ)

    wp = FullLeasedProblem()

    u,v = randn(num_i), randn(num_i)

    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, royalty_rates)
    royrates_endog = observed_choices(data_roy)
    @test royalty_rates[_y(data_roy)] == royrates_endog
    royrates_exog = sample(royalty_rates, num_i)

    _ichars = SDLParameters.ichars(log_ogip, royrates_endog)
    @test _ichars == [gr for gr in zip(log_ogip, royrates_endog)]

    # model
    ddm_no_t1ev     = TestDynamicDrillModel(price_process, false; reward=rwrd_u)
    ddm_with_t1ev   = TestDynamicDrillModel(price_process, true;  reward=rwrd_u)
    ddm_c_no_t1ev   = TestDynamicDrillModel(price_process, false; reward=rwrd_c)
    ddm_c_with_t1ev = TestDynamicDrillModel(price_process, true;  reward=rwrd_c)

    # construct drilling data
    ddm_opts = (minmaxleases=1:1, nper_initial=40:40, tstart=1:50)
    data_drill_w = DataDynamicDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    data_drill_n = DataDynamicDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)

    # constrained versions of above
    data_drill_w_con = DataDrill(ddm_c_with_t1ev, data_drill_w)
    data_drill_n_con = DataDrill(ddm_c_no_t1ev, data_drill_n)

    # number of wells drilled
    nwells_w = total_wells_drilled(data_drill_w)
    nwells_n = total_wells_drilled(data_drill_n)
    @test nwells_w == map(s -> _D(wp,s), max_states(data_drill_w))
    @test nwells_n == map(s -> _D(wp,s), max_states(data_drill_n))

    data_produce_w = DataProduce(u, nwells_w, obs_per_well, θ_produce, log_ogip)
    data_produce_n = DataProduce(u, nwells_n, obs_per_well, θ_produce, log_ogip)

    data_d_sm = data_drill_w_con
    data_d_lg = data_drill_w
    data_p    = data_produce_w
    data_r    = data_roy
    empty = EmptyDataSet()

    coef_links = CoefLinks(ddm_no_t1ev)
    @test coef_links == [(idx_produce_ψ, idx_drill_ψ,), (idx_produce_g, idx_drill_g)]

    data_dril = DataSetofSets(data_d_sm, empty, empty)
    data_full = DataSetofSets(data_d_lg, data_r, data_p, CoefLinks(data_d_lg))

    theta_dril = θ_drill_c
    theta_full = merge_thetas((θ_drill_u, θ_royalty, θ_produce), data_full)


end # test
end # module
