module SDLParameters

using Distributions
using ShaleDrillingLikelihood
using Dates
using SparseArrays
using LinearAlgebra: checksquare

using ShaleDrillingLikelihood: simulate,
    cost, extend, revenue,
    lrmean, lrvar, lrstd,
    zero_out_small_probs,
    extra_parm

import ShaleDrillingLikelihood: ichars

using Base: product

export NormalOGIP, logOGIP,
    PriceProcess, TestPriceProcess,
    LeasedProblem,
    RoyaltyRates,
    RoyaltyExogVars,
    RealDiscountRate,
    PsiSpace,
    TestPsiSpace,
    ichars,
    TestDynamicDrillModel

# ----------------------------
# Geology
# ----------------------------

NormalOGIP(;mu=4.68, sigma=0.31) = Normal(mu,sigma)

logOGIP(num_i, d=NormalOGIP()) = rand(d, num_i)

# ----------------------------
# Price process
# ----------------------------

struct PriceProcess{TS<:TimeSeriesProcess, SRL<:Tuple, ETV<:ExogTimeVars, M<:AbstractMatrix}
    process::TS
    grid::SRL
    series::ETV
    transition::M

    function PriceProcess(process::TS, grid::SRL, series::ETV, transition::M) where {TS,SRL,ETV,M}
        all(minimum.(grid) .<= minimum(series)) || throw(error())
        all(maximum.(grid) .>= maximum(series)) || throw(error())
        length(product(grid...)) == checksquare(transition) || throw(DimensionMismatch())
        new{TS,SRL,ETV,M}(process,grid,series,transition)
    end
end

process(   x::PriceProcess) = x.process
grid(      x::PriceProcess) = x.grid
series(    x::PriceProcess) = x.series
transition(x::PriceProcess) = x.transition

function TestAR1Process(;rho=0.8, mean=1.33*(1-rho), var=0.265^2*(1-rho^2))
    return AR1process(mean, rho, var)
end

function TestPriceDates(n=150;start=Date(2003,10), step=Quarter(1))
    return range(start; step=step, length=n)
end

function TestPriceGrid(zvec, length=15; process=TestPriceProcess())
    zmin = min(minimum(zvec), lrmean(process)-3*lrstd(process))
    zmax = max(maximum(zvec), lrmean(process)+3*lrstd(process))
    zrng = range(zmin, zmax; length=length)
    return tuple(zrng)
end

function TestPriceTransition(grid::StepRangeLen; process=TestPriceProcess(), minp=1e-4)
    Pdense = tauchen_1d(process, grid)
    Pdense_zero = zero_out_small_probs(Pdense, minp)
    return sparse(Pdense_zero)
end

function TestPriceProcess(; process=TestAR1Process(), nsim=150, ngrid=13)
    dates = TestPriceDates(nsim)
    series = simulate(process, nsim)
    etv = ExogTimeVars(tuple.(series), dates)
    grid = TestPriceGrid(series, ngrid; process=process)
    transition = TestPriceTransition(grid...; process=process)
    return PriceProcess(process, grid, etv, transition)
end

# ----------------------------
# Lease problem
# ----------------------------

function LeasedProblem(s::Symbol=:full)
    if s == :small
        lp = LeasedProblem(3, 4, 5, 3, 2)
    elseif s == :full
        lp = LeasedProblem(8, 8, 12, 12, 8)
    else
        throw(error("don't konw s = $s"))
    end
    return lp::LeasedProblem
end

# ----------------------------
# Royalty rates
# ----------------------------

function RoyaltyRates(n::Integer)
    if n == 3
        rr = [1/8, 3/16, 1/4,]
    elseif n == 6
        rr = [1/8, 1/6, 3/16, 1/5, 9/40, 1/4]
    else
        throw(error("don't know this n=$n"))
    end
    return rr::Vector{Float64}
end

function RoyaltyRates(s::Symbol=:full)
    if s == :small
        rr = RoyaltyRates(3)
    elseif s == :full
        rr = RoyaltyRates(6)
    else
        throw(error("don't konw s = $s"))
    end
    return rr
end

function RoyaltyExogVars(num_i::Integer, rates, theta, log_ogip=zeros(num_i,0), model=RoyaltyModel())
    L = length(rates)
    k = length(theta) - (L-1) - extra_parm(model)
    @assert k > 0
    kminus = k - size(log_ogip',1)
    @assert kminus > 0
    Xminus = randn(kminus, num_i)
    X = vcat(log_ogip', Xminus)
    return X
end

# ----------------------------
# Model
# ----------------------------

RealDiscountRate() = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)
PsiSpace(n; start=-4.5, stop=4.5) = range(start; stop=stop, length=n)
TestPsiSpace() = PsiSpace(13)

ichars(log_ogip, royalty_rates) = collect(zip(log_ogip, royalty_rates))

function TestDynamicDrillModel(z::PriceProcess; reward = UnconstrainedProblem,
    discount = RealDiscountRate(),
    statespace = LeasedProblem(:full),
    psispace = TestPsiSpace(),
    anticipate_t1ev = true
)

    f = DrillReward(
        DrillingRevenue(Constrained(),NoTrend(),NoTaxes()),
        DrillingCost_constant(),
        ExtensionCost_Constant()
    )
    rwrd = reward(f)
    zs = tuple(grid(z))
    ztrans = transition(z)
    ddm = DynamicDrillModel(
        rwrd, discount, statespace, zs, ztrans, psispace, anticipate_t1ev
    )
    return ddm
end

# ----------------------------
# Parameters
# ----------------------------

export ThetaRho,
    AlphaPsi, AlphaG, AlphaT, Alpha0,
    Theta

ThetaRho() = 0.0
AlphaPsi() = 0.33
AlphaG() = 0.56
AlphaT() = 0.02
Alpha0() = -2.8
# θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
# #            drill  ext    α0  αg  αψ  θρ
# θ_drill_u = [-5.5, -2.0, -2.8, αg, αψ, θρ]
# # θ_drill_u = [-5.8,       -2.8, αg, αψ, θρ]
# θ_drill_c = vcat(θ_drill_u[1:3], θρ)
#

Theta(m, args...) = throw(error("Default Theta not assigned for $m"))
Theta(m::AbstractDynamicDrillModel, args...) = Theta(reward(m), args...)
function Theta(m::DrillReward, args...)
    c = Theta(cost(m), args...)
    e = Theta(extend(m), args...)
    r = Theta(revenue(m), args...)
    return vcat(c,e,r)
end

# Royalty
function Theta(m::RoyaltyModel; θρ=ThetaRho(), kwargs...)
    #            θρ  ψ        g    x_r      κ₁   κ₂
    return vcat(θρ, 1.0,    0.1, -0.3,    -0.6, 0.6)
end

# Production
function Theta(m::ProductionModel; αψ=AlphaPsi(), αg=AlphaG(), kwargs...)
    #           αψ, αg, γx   σ2η, σ2u  # NOTE: η is iwt, u is iw
    return vcat(αψ, αg, 0.2, 0.3, 0.4)
end

# Test Drill
function Theta(m::TestDrillModel; θρ=ThetaRho(), αψ=AlphaPsi(), kwargs...)
    return vcat(αψ, 2.0, -2.0, -0.75, θρ)
end


# revenue
# ---------
function Theta(m::DrillingRevenue{Constrained}; θρ=ThetaRho(), kwargs...)
    return vcat(Alpha0(), θρ)
end

function Theta(m::DrillingRevenue{Unconstrained, NoTrend};
    θρ=ThetaRho(), αψ=AlphaPsi(), αg=AlphaG(), α0=Alpha0(), kwargs...)
    return vcat(α0, αg, αψ, θρ)
end

function Theta(m::DrillingRevenue{Unconstrained, TimeTrend};
    θρ=ThetaRho(), αψ=AlphaPsi(), αg=AlphaG(), αT=AlphaT(), α0=Alpha0(), kwargs...)
    return vcat(α0, αg, αψ, αt, θρ)
end

# cost
# ---------
Theta(m::DrillingCost_constant, args...) = vcat(-5.5)

# extension
# ---------
Theta(m::ExtensionCost_Zero    , args...) = zeros(0)
Theta(m::ExtensionCost_Constant, args...) = vcat(-2.0)









end
