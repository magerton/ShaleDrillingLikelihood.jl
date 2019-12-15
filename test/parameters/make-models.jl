module ShaleDrillingLikelihood_MakeModels

using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using Test
using SparseArrays
using Random

using ShaleDrillingLikelihood: DataDynamicDrill, total_wells_drilled
using ShaleDrillingLikelihood.SDLParameters: series

println("print to keep from blowing up")

@testset "Parameters" begin
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=AlphaG(), α_ψ=AlphaPsi()),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u

    θ_royalty = Theta(RoyaltyModel())
    θ_produce = Theta(ProductionModel())
    θ_drill_c = Theta(rwrd_c)
    θ_drill_u = Theta(rwrd_u)

    num_zt = 150          # drilling
    nz = 15
    num_i = 50
    nψ =  13
    M = 50
    obs_per_well = 10:20  # pdxn

    # grids
    ψs = PsiSpace(nψ)
    wp = FullLeasedProblem()
    royalty_rates = RoyaltyRatesSmall()

    # make RVs
    Random.seed!(12345)
    u,v = randn(num_i), randn(num_i)
    z = TestPriceProcess(;nsim=num_zt, ngrid=nz)
    _zchars = series(z)
    log_ogip = logOGIP(num_i)

    # construct royalty data
    data_roy = TestDataRoyalty(u, v, royalty_rates, θ_royalty, log_ogip)
    _ichars = SDLParameters.ichars(log_ogip, observed_choices(data_roy))

    # models
    ddm_no_t1ev     = TestDynamicDrillModel(z, false; problem=UnconstrainedProblem)
    ddm_with_t1ev   = TestDynamicDrillModel(z, true;  problem=UnconstrainedProblem)
    ddm_c_no_t1ev   = TestDynamicDrillModel(z, false; problem=ConstrainedProblem)
    ddm_c_with_t1ev = TestDynamicDrillModel(z, true;  problem=ConstrainedProblem)

    # construct drilling data
    ddm_opts = (minmaxleases=1:1, nper_initial=40:40, tstart=1:50)
    data_drill_w = DataDynamicDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    data_drill_n = DataDynamicDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)

    # number of wells drilled
    data_produce_w = TestDataProduce(u, data_drill_w, obs_per_well, θ_produce, log_ogip)
    data_produce_n = TestDataProduce(u, data_drill_n, obs_per_well, θ_produce, log_ogip)

    # constrained versions of above
    data_d_sm = DataDrill(ddm_c_with_t1ev, data_drill_w)
    data_d_lg = data_drill_w
    data_p    = data_produce_w
    data_r    = data_roy
    empty = EmptyDataSet()

    data_dril = DataSetofSets(data_d_sm, empty, empty)
    data_full = DataSetofSets(data_d_lg, data_r, data_p, CoefLinks(data_d_lg))

    theta_dril = θ_drill_c
    theta_full = merge_thetas((θ_drill_u, θ_royalty, θ_produce), data_full)


end # test
end # module
