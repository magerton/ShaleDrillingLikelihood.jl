module ShaleDrillingLikelihood_DynamicDrillModel_ComparativeStatics

DOBTIME = false
const DOPROFILE = false

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
# using Profile
# using PProf
# using Juno
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
    j2ptr, tptr


println("print to keep from blowing up")

SEED = 1234

@testset "simulate all data" begin

    Random.seed!(SEED)

    discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

    # set up coefs
    θρ = 0.0
    αψ = 0.33
    αg = 0.56

    # set up coefs
    #            θρ  ψ        g    x_r     κ₁   κ₂
    θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
    #                αψ, αg, γx   σ2η, σ2u   # η is iwt, u is iw
    θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)
    #            drill  ext    α0  αg  αψ  θρ

    # model
    rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=αg, α_ψ=αψ),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    rwrd_u = UnconstrainedProblem(rwrd_c)
    rwrd = rwrd_u

    # parms
    num_i = 250

    # grid sizes
    nψ =  13
    nz = 15

    # simulations
    M = 250

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

    # construct royalty data
    data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)
    royrates_endog = royalty_rates[_y(data_roy)]
    royrates_exog = sample(royalty_rates, num_i)

    # model
    ddm = DynamicDrillModel(rwrd_u, discount, wp, zs, ztrans, ψs, false)
    ddm_opts = (minmaxleases=0:0, nper_initial=0:0, nper_development=60:60, tstart=1:75, xdomain=0:0)

    theta1  = [-5.5, -2.0, -2.8, αg, αψ, θρ]
    theta2  = [-4.5, -2.0, -2.8, αg, αψ, θρ]
    theta3  = [-5.5, -1.0, -2.8, αg, αψ, θρ]
    theta4  = [-5.5, -2.0, -2.5, αg, αψ, θρ]
    theta5  = [-5.5, -2.0, -2.8, αg+0.1, αψ, θρ]
    theta6  = [-5.5, -2.0, -2.8, αg, αψ+0.3, θρ]
    theta7  = [-5.5, -2.0, -2.8, αg, αψ, θρ+0.5]

    thetas = (theta1,theta2,theta3,theta4,theta5,theta6,theta7)

    @test _nparm(rwrd) == length(theta1)

    # construct ichars for drilling
    _ichars = [gr for gr in zip(log_ogip, royrates_exog)]

    function set_seed_and_data(thet)
        Random.seed!(SEED)
        return DataDrill(u, v, _zchars, _ichars, ddm, thet; ddm_opts...)
    end

    function last_state_dist(data::DataDrill)
        x = _x(data)
        _tptr = tptr(data)
        _j2ptr = j2ptr(data)
        first_tptr = _j2ptr[2]
        last_obs = _tptr[first_tptr:end] .- 1
        last_x = x[ last_obs ]
        return sort(countmap(last_x))
    end

    # using Juno
    # Juno.@enter DataDrill(u, v, _zchars, _ichars, ddm, theta1; ddm_opts...)

    datas = map(set_seed_and_data, thetas)
    @show ys = map(d -> sum(_y(d)), datas)
    xterms = map(last_state_dist, datas)


    for x in xterms
        @show x
    end

    @test ys[1] < ys[2] # cost down ⟹ drill up
    @test ys[1] > ys[3] # ext cost down ⟹ drill down
    @test ys[1] < ys[4] # α₀ up ⟹ drill up
    @test ys[1] < ys[5] # αg up ⟹ drill up b/c E(gᵢ) > 0
    @test ys[1] < ys[6] # αψ up ⟹ more exploration, more exhaustion

    expired = end_ex0(wp) + 1
    exhausted = end_inf(wp)
    @test xterms[1][expired] < xterms[3][expired] # cheap expiration ⟹ more expire

    @test xterms[1][expired] > xterms[6][expired] # αψ ↑ ⟹ LESS expired
    @test xterms[1][exhausted] < xterms[6][exhausted] # αψ ↑ ⟹ MORE exhausted

    @show xterms[1][expired]   < xterms[7][expired] # ρ ↑ ⟹ MORE expired
    @show xterms[1][exhausted] , xterms[7][exhausted] # ρ ↑ ⟹ MORE exhausted


    #TODO: test payoff(d=0) == 0 ∀ sidx ≠ end_ex1(wp)

    # # states
    # end_ex1(wp), end_ex0(wp)
    #
    # sort(countmap(_x(data_drill_n_con)))
    # # count of where we are before dirlling first
    # # frequency of choices before expiration
    # sort(countmap([x for (x,y) in zip(_x(data_drill_n_con), _y(data_drill_n_con)) if x <= end_ex0(wp) && y > 0]))
    #
    # sort(countmap(_x(data_drill_n_con)[tptr(data_drill_n_con)[j2ptr(data_drill_n_con)[2]:end].-1]))
    #
    # # frequency of choices before expiration
    # sort(countmap([y for (x,y) in zip(_x(data_drill_n_con), _y(data_drill_n_con)) if x <= end_ex0(wp)]))
    # # infill drilling choices
    # sort(countmap([y for (x,y) in zip(_x(data_drill_n_con), _y(data_drill_n_con)) if x > end_lrn(wp)]))
    # # frequency of states just before where D==2 at end_lrn(wp)+2
    # sort(countmap(_x(data_drill_n_con)[findall(x -> x==end_lrn(wp)+2, _x(data_drill_n_con)).-1]))

end





end # module
