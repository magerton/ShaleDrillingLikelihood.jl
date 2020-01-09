using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Test
using SparseArrays
using BenchmarkTools
using Profile
# using ProfileView
using InteractiveUtils

using Base.Iterators: product, OneTo

using AxisAlgorithms: A_mul_B_md!
using ShaleDrillingLikelihood: beta_1minusbeta, update_IminusTVp!, IminusTEVp
using LinearAlgebra
using IterativeSolvers
using IncompleteLU

using ShaleDrillingLikelihood: DCDPTmpVars,
    q,
    lse,
    tmp,
    ubV,
    tmp_cart,
    Πψtmp,
    IminusTEVp,
    ubVfull,
    reward,
    dp1space,
    sprimes,
    vfit!,
    dEV,
    EV,
    anticipate_t1ev,
    _dmax,
    solve_inf_vfit!,
    solve_inf_pfit!,
    solve_inf_vfit_pfit!,
    gradinf!,
    update_static_payoffs!

nz = 51
nψ = 51
num_zt = 15

royalty_rates = RoyaltyRatesSmall()
rwrd    = DefaultDrillReward()
θ_drill   = Theta(DefaultDrillReward())
wp = FullLeasedProblem()
anticipate = true
seed=1234

z = TestPriceProcess(;nsim=num_zt, ngrid=nz)
_zchars = ShaleDrillingLikelihood.SDLParameters.series(z)

ddm  = TestDynamicDrillModel(z, anticipate; reward=rwrd)
tmpv = DCDPTmpVars(ddm)
evs = ValueFunction(ddm)

itype = (4.67, 0.25,)
dograd = false

i = length(wp) - 1 # state_index

idxd  = dp1space(wp,i)
idxs  = collect(sprimes(statespace(ddm),i))

EV0   = view(EV(evs) ,    :, :, i)
EV1   = view(EV(evs) ,    :, :, idxs)
dEV0  = view(dEV(evs), :, :, :, i)
dEV1  = view(dEV(evs), :, :, :, idxs)

fill!(evs, 0)

tmp_vw = view(tmpv, idxd)
update_static_payoffs!(tmp_vw, ddm, θ_drill, i, itype, dograd)

ubV(tmp_vw)  .+= discount(ddm) .* EV1

converged, iter, bnds = solve_inf_vfit!(EV0, tmp_vw, ddm; maxit=10, vftol=1e-5)

let t = tmp_vw
    bnds = ShaleDrillingLikelihood.pfit!(EV0, t, ddm; vftol=1e-10)
    ubVfull(t)[:,:,1] .= discount(ddm) .* EV0
end


t = tmp_vw
vftol = 1e-16

ΔEV = lse(t)
q0 = ubV(t)

if anticipate_t1ev(ddm)
    logsumexp_and_softmax!(ΔEV, q0, tmp(t), ubV(t), 1)
else
    findmax!(add_1_dim(ΔEV), add_1_dim(tmp_cart(t)), ubV(t))
    q0[:,:,1] .= last.(getfield.(tmp_cart(t), :I)) .== 1         # update q0 as Pr(d=0|x)
end
A_mul_B_md!(tmp(t), ztransition(ddm), ΔEV, 1)

# compute difference & check bnds
bnds = extrema(ΔEV .= EV0 .- tmp(t)) .* -beta_1minusbeta(ddm)
if all(abs.(bnds) .< vftol)
    EV0 .= tmp(t)
    return bnds
end

j = 1
q0j  = view(q0, :, j, 1)
ΔEVj_orig = view(ΔEV, :, j)
ΔEVj = similar(ΔEVj_orig)

# Consider https://juliamath.github.io/IterativeSolvers.jl/dev/preconditioning/#Preconditioning-1
# with https://github.com/haampie/IncompleteLU.jl as preconditioner
# https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
# https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
# Econ paper - https://doi.org/10.1016/S1474-6670(17)33089-6
# Mrkaic and Pauletto (2001) Preconditioning in Economic Stochastic Growth Models
update_IminusTVp!(t, ddm, q0j)

function solvedirect(t, ΔEVj, ΔEVj_orig)
    ΔEVj .= IminusTEVp(t) \ ΔEVj_orig
end

using Preconditioners

function solveindirect(t, ΔEVj, ΔEVj_orig; kwargs...)
    # bicgstabl!(ΔEVj, IminusTEVp(t), ΔEVj_orig; kwargs...)
    gmres!(ΔEVj, IminusTEVp(t), ΔEVj_orig; kwargs...)
end

function solveindirect_plu(t, ΔEVj, ΔEVj_orig, tau = 0.02; kwargs...)
    LU = ilu(IminusTEVp(t); τ = tau)
    # bicgstabl!(ΔEVj, IminusTEVp(t), ΔEVj_orig; Pl=LU, kwargs...)
    gmres!(ΔEVj, IminusTEVp(t), ΔEVj_orig; Pl=LU, kwargs...)
end

function solveindirect_diag(t, ΔEVj, ΔEVj_orig, dp::DiagonalPreconditioner; kwargs...)
    A = IminusTEVp(t)
    # dp = DiagonalPreconditioner(A)
    UpdatePreconditioner!(dp, A)
    # bicgstabl!(ΔEVj, A, ΔEVj_orig; Pl=LU, kwargs...)
    gmres!(ΔEVj, A, ΔEVj_orig; Pl=dp, kwargs...)
end

dp = DiagonalPreconditioner(IminusTEVp(t))

tol = 1e-13
@btime solvedirect(  $t, $ΔEVj, $ΔEVj_orig)
@btime solveindirect($t, $ΔEVj, $ΔEVj_orig; tol=tol)
@btime solveindirect_plu($t, $ΔEVj, $ΔEVj_orig; tol=tol)
@btime solveindirect_diag($t, $ΔEVj, $ΔEVj_orig, $dp; tol=tol)

A = IminusTEVp(t)
ILU = ilu(A; τ = 1e-4)
LU = lu(A)
@btime ilu($A; τ = 1e-4)
@btime lu($A)
@btime gmres!($ΔEVj, $A, $ΔEVj_orig; Pl=$ILU)
@btime solvedirect(  $t, $ΔEVj, $ΔEVj_orig)

# for tau in (2.0, 1.0, 0.1, 0.01, 0.001)
#     @btime solveindirect_plu($t, $ΔEVj, $ΔEVj_orig, $tau)
# end


bicgstabl!(ΔEVj, IminusTEVp(t), ΔEVj_orig; log=true)

DiagonalPreconditioner(IminusTEVp(t))

for tau in (1.0, 0.1, 0.02, 0.01, 0.001)
    LU = ilu(IminusTEVp(t); τ = tau)
    xplu, hist = gmres!(ΔEVj, IminusTEVp(t), ΔEVj_orig; Pl=LU, log=true)
    @show hist
end

x, hist = gmres!(ΔEVj, IminusTEVp(t), ΔEVj_orig; log=true)
@show hist

UpdatePreconditioner!(dp, IminusTEVp(t))
x, hist = gmres!(ΔEVj, IminusTEVp(t), ΔEVj_orig; Pl=dp, log=true)
@show hist






















# converged, iterp, bnds = solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-10, maxit0=20, maxit1=40)
