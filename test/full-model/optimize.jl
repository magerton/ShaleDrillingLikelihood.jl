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
    EV, dEV,
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
    parallel_simloglik!


println("print to keep from blowing up")


@testset "simulate all data" begin

    Random.seed!(7)

    discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

    # set up coefs
    θρ =  0.0
    αψ = 0.33
    αg = 0.56

    # set up coefs
    #            θρ  ψ        g    x_r     κ₁   κ₂
    θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
    #                αψ, αg, γx   σ2η, σ2u   # η is iwt, u is iw
    θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)
    #            drill  ext    α0  αg  αψ  θρ
    θ_drill_u = [-5.5, -2.0, -2.8, αg, αψ, θρ]
    # θ_drill_u = [-5.8,       -2.8, αg, αψ, θρ]
    θ_drill_c = vcat(θ_drill_u[1:3], θρ)

    # model
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=αg, α_ψ=αψ),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u
    @test _nparm(rwrd) == length(θ_drill_u)

    # parameters
    num_i = 3

    # grid sizes
    nψ =  13
    nz = 15

    # simulations
    M = 5

    # observations
    num_zt = 150          # drilling
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

    # model
    ddm_no_t1ev     = DynamicDrillModel(rwrd_u, discount, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev   = DynamicDrillModel(rwrd_u, discount, wp, zs, ztrans, ψs, true)
    ddm_c_no_t1ev   = DynamicDrillModel(rwrd_c, discount, wp, zs, ztrans, ψs, false)
    ddm_c_with_t1ev = DynamicDrillModel(rwrd_c, discount, wp, zs, ztrans, ψs, true)

    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)
    royrates_endog = royalty_rates[_y(data_roy)]
    royrates_exog = sample(royalty_rates, num_i)

    # construct ichars for drilling
    _ichars = [gr for gr in zip(log_ogip, royrates_exog)]

    # construct drilling data
    # ddm_opts = (minmaxleases=1:1, nper_initial=40:40, tstart=1:50)
    # data_drill_w = DataDynamicDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)
    # data_drill_n = DataDynamicDrill(u, v, _zchars, _ichars, ddm_no_t1ev,   θ_drill_u; ddm_opts...)
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
    data_royp = DataSetofSets(empty, data_r, data_p)

    theta_dril = θ_drill_c
    theta_tuple = (θ_drill_u, θ_royalty, θ_produce)
    theta_full = merge_thetas(theta_tuple, data_full)
    theta_royp = vcat(θ_royalty, θ_produce)


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

    for (d,t) in zip( [data_royp, data_dril, data_full], [theta_royp, theta_dril, theta_full] )

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

        # leograd = ShaleDrillingLikelihood.grad(leo)
        #
        # fill!(leograd, 0)
        # serial_simloglik!(ew, t, true)
        # gs = copy(leograd)
        # @test any(gs .!= 0)
        #
        # fill!(leograd, 0)
        # parallel_simloglik!(ew, t, true)
        # gp = copy(leograd)
        # @test any(gp .!= 0)
        #
        # @test gs ≈ gp
        #
        # fds = Calculus.gradient(x -> serial_simloglik!(  ew, x, false), t)
        # fdp = Calculus.gradient(x -> parallel_simloglik!(ew, x, false), t)
        # @test isapprox(gs , fds; atol=4e-6, rtol=10*sqrt(eps(eltype(gs))))
        # @test isapprox(fds, fdp; atol=4e-6, rtol=10*sqrt(eps(eltype(gs))))
        # @test isapprox(gp,  fdp; atol=4e-6, rtol=10*sqrt(eps(eltype(gs))))

    end


    end


































end # module
