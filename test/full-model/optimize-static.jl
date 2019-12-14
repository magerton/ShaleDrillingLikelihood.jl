module ShaleDrillingLikelihood_Optimize

DOBTIME = false
DOPROFILE = false

using ShaleDrillingLikelihood
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
    _x, _y,
    end_ex0, end_ex1, end_lrn, end_inf,
    state_idx,
    j2ptr, tptr,
    idx_produce_ψ, idx_drill_ψ,
    idx_produce_g, idx_drill_g,
    merge_thetas,
    split_thetas,
    theta_produce,
    serial_simloglik!,
    parallel_simloglik!,
    TestDrillModel



println("print to keep from blowing up")


@testset "simulate all data" begin

    Random.seed!(7)

    discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

    # set up coefs
    θρ =  0.0
    αψ = 0.33
    αg = 0.56

    model_drill = TestDrillModel()
    θ_drill   = [αψ, 2.0, -2.0, -0.75, θρ]
    θ_royalty = [θρ, 1.0,    -1.0, 1.0, 1.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ
    θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)

    θ = vcat(θ_drill, θ_royalty[2:end], θ_produce[2:end])

    M = 250
    num_i = 500
    num_zt = 150          # drilling

    # observations
    obs_per_well = 10:20  # pdxn

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

    # simulate price process
    zprocess = AR1process(zmean, zrho, zvar)
    zvec = simulate(zprocess, num_zt)
    ztrng = range(Date(2003,10); step=Quarter(1), length=num_zt)
    _zchars = ExogTimeVars(tuple.(zvec), ztrng)

    # random vars
    log_ogip = rand(ogip_dist, num_i)
    Xroyalty = vcat(log_ogip', randn(nk_royalty-1, num_i))
    u,v = randn(num_i), randn(num_i)

    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)
    royrates_endog = royalty_rates[_y(data_roy)]
    royrates_exog = sample(royalty_rates, num_i)

    # construct ichars for drilling
    _ichars = [gr for gr in zip(log_ogip, data_roy)]


    data_drill_opt = (minmaxleases=1:2, nper_initial=10:20, nper_development=0:10, tstart=1:50,)
    data_drill   = DataDrill(u, v, _zchars, _ichars, TestDrillModel(), θ_drill; data_drill_opt...)
    data_produce = DataProduce(u, 10, obs_per_well, θ_produce, log_ogip)

    coef_links = [(idx_produce_ψ, idx_drill_ψ,),]
    data_full = DataSetofSets(data_drill, data_roy, data_produce, coef_links)
    data_small = DataSetofSets(EmptyDataSet(), EmptyDataSet(), data_produce)


    DOPAR = false

    if DOPAR
        rmprocs(workers())
        pids = addprocs()
    else
        pids = [1,]
    end
    println_time_flush("Putting pkg on workers")
    @everywhere using ShaleDrillingLikelihood
    println_time_flush("Package on workers")


    # d = data_full
    # t = θ
    d = data_small
    t = θ_produce

    resetcount!()

    leo = LocalEstObj(d,t)
    s = SimulationDraws(M, ShaleDrillingLikelihood.data(leo))
    reo = RemoteEstObj(leo, M)
    ew = EstimationWrapper(leo, reo)

    for dograd in false:true
        @eval @everywhere set_g_RemoteEstObj($reo)
        simloglik!(1, t, dograd, reo)
        serial_simloglik!(ew, t, dograd)
        parallel_simloglik!(ew, t, dograd)
    end

    # startcount!([1, 100000,], [100, 100,])
    resetcount!()
    startcount!([100, 200, 500, 100000,], [1, 5, 100, 100,])
    res = solve_model(ew, t; show_trace=true, time_limit=10)
    @show res
    @test minimizer(res) == theta1(leo)

end


































end # module
