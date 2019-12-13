using ShaleDrillingLikelihood
using SparseArrays
using Random
using Distributions
using Dates
using StatsBase
using LinearAlgebra
using JLD2

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
    j2ptr, tptr, j1ptr, ichars, zchars


Random.seed!(7)

discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

# set up coefs
θρ = -4.0
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
θ_drill_c = vcat(θ_drill_u[1:2], θρ)

# model
rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=αg, α_ψ=αψ),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
rwrd_u = UnconstrainedProblem(rwrd_c)
rwrd = rwrd_u

# parms
num_i = 350

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
ztrans = sparse(zero_out_small_probs(tauchen_1d(zprocess, zrng), 1e-4))
@show zrng

# random vars
log_ogip = rand(ogip_dist, num_i)
Xroyalty = vcat(log_ogip', randn(nk_royalty-1, num_i))
u,v = randn(num_i), randn(num_i)

# model
ddm_with_t1ev = DynamicDrillingModel(rwrd_u, discount, wp, zs, ztrans, ψs, true)

# construct royalty data
data_roy = DataRoyalty(u, v, Xroyalty, θ_royalty, num_royalty_rates)
royrates_endog = royalty_rates[_y(data_roy)]
royrates_exog = sample(royalty_rates, num_i)

# construct ichars for drilling
_ichars = [gr for gr in zip(log_ogip, royrates_exog)]

ddm_opts = (minmaxleases=0:0, nper_initial=0:0, nper_development=60:60, tstart=1:75, xdomain=0:0)
data = DataDrill(u, v, _zchars, _ichars, ddm_with_t1ev, θ_drill_u; ddm_opts...)

modulepath = dirname(pathof(ShaleDrillingLikelihood))
filepath = joinpath(modulepath, "..", "scratch", "simulated-data.jld2")

jldopen(filepath, "w") do file
    file["j1ptr"]   = j1ptr(data)
    file["j2ptr"]   = j2ptr(data)
    file["tptr"]    = tptr(data)
    file["jtstart"] = jtstart(data)
    file["j1chars"] = j1chars(data)
    file["ichars"]  = ichars(data)
    file["y"]       = _y(data)
    file["x"]       = _x(data)
    file["zchars"]  = zchars(data).timevars
    file["zrng"] = zrng
    file["psis"] = ψs
    file["ztrans"] = ztrans
end
