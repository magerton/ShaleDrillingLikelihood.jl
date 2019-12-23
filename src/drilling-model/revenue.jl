export AbstractTaxType,
    AbstractTechChange,
    AbstractConstrainedType,
    AbstractLearningType,
    AbstractRoyaltyType,
    NoTaxes,
    WithTaxes,
    GathProcess,
    NoTrend,
    TimeTrend,
    Unconstrained,
    Constrained,
    DrillingRevenue,
    Learn,
    NoLearn,
    WithRoyalty,
    NoRoyalty,
    ConstrainedProblem,
    UnconstrainedProblem,
    NoLearningProblem,
    STARTING_α_ψ,
    STARTING_log_ogip,
    STARTING_α_t

# ----------------------------------------------------------------
# Drilling revenue variations
# ----------------------------------------------------------------

abstract type AbstractLearningType    <: AbstractModelVariations end
abstract type AbstractConstrainedType <: AbstractModelVariations end
abstract type AbstractTechChange      <: AbstractModelVariations end
abstract type AbstractTaxType         <: AbstractModelVariations end
abstract type AbstractRoyaltyType     <: AbstractModelVariations end

# drilling revenue
struct DrillingRevenue{
    Cn <: AbstractConstrainedType, Tech <: AbstractTechChange,
    Tax <: AbstractTaxType, Lrn <: AbstractLearningType,
     Roy <: AbstractRoyaltyType
} <: AbstractDrillingRevenue
    constr::Cn
    tech::Tech
    tax::Tax
    learn::Lrn
    royalty::Roy
end

DrillingRevenue(Cn, Tech, Tax) = DrillingRevenue(Cn, Tech, Tax, Learn(), WithRoyalty())

@inline constr( x::DrillingRevenue) = x.constr
@inline tech(   x::DrillingRevenue) = x.tech
@inline tax(    x::DrillingRevenue) = x.tax
@inline learn(  x::DrillingRevenue) = x.learn
@inline royalty(x::DrillingRevenue) = x.royalty

# Constrained parameters
# -----------------------

# Whether we've constrained coefs
struct Unconstrained <: AbstractConstrainedType end
struct Constrained   <: AbstractConstrainedType
    log_ogip::Float64
    α_ψ::Float64
    α_t::Float64
end
Constrained(; log_ogip=STARTING_log_ogip, α_ψ = STARTING_α_ψ, α_t = STARTING_α_t, args...) = Constrained(log_ogip, α_ψ, α_t)

@inline theta_g(x::Constrained) = x.log_ogip
@inline theta_ψ(x::Constrained) = x.α_ψ
@inline theta_t(x::Constrained) = x.α_t

const DrillingRevenueConstrained = DrillingRevenue{Constrained}
const DrillingRevenueUnconstrained = DrillingRevenue{Unconstrained}

# Technology
# -----------------

const TIME_TREND_BASE = 2008

struct NoTrend <: AbstractTechChange  end
struct TimeTrend <: AbstractTechChange
    baseyear::Int
end
TimeTrend() = TimeTrend(TIME_TREND_BASE)
@inline baseyear(x::TimeTrend) = x.baseyear
function baseyear(x::NoTrend)
    yr = TIME_TREND_BASE
    # @warn "No base year defined for $x. using $yr"
    return yr
end
baseyear(x::DrillingRevenue) = baseyear(tech(x))
baseyear(x::DrillReward) = baseyear(revenue(x))

const DrillingRevenueTimeTrend = DrillingRevenue{Cn,TimeTrend} where {Cn}
const DrillingRevenueNoTrend   = DrillingRevenue{Cn,NoTrend} where {Cn}

@inline centered_time(x::DrillingRevenue, z::Tuple) = last(z) - baseyear(tech(x))
@inline trend_component(x::DrillingRevenueTimeTrend, θ, z) = theta_t(x,θ) * centered_time(x, z)
@inline trend_component(x::DrillingRevenueNoTrend, θ, z) = 0

# chebshev polynomials
# See http://www.aip.de/groups/soe/local/numres/bookcpdf/c5-8.pdf
@inline checkinterval(x::Real,min::Real,max::Real) =  min <= x <= max || throw(DomainError("x = $x must be in [$min,$max]"))
@inline checkinterval(x::Real) = checkinterval(x,-1,1)

@inline cheb0(z::Real) = one(z)
@inline cheb1(z::Real) = clamp(z,-1,1)
@inline cheb2(z::Real) = (x = cheb1(z); return 2*x^2 - 1)
@inline cheb3(z::Real) = (x = cheb1(z); return 4*x^3 - 3*x)
@inline cheb4(z::Real) = (x = cheb1(z); return 8*(x^4 - x^2) + 1)

# learning
# -----------------------

struct Learn       <: AbstractLearningType end
struct NoLearn     <: AbstractLearningType end
struct PerfectInfo <: AbstractLearningType end
struct MaxLearning <: AbstractLearningType end

const DrillingRevenueLearn       = DrillingRevenue{Cn,Tech,Tax,Learn}       where {Cn,Tech,Tax}
const DrillingRevenueNoLearn     = DrillingRevenue{Cn,Tech,Tax,NoLearn}     where {Cn,Tech,Tax}
const DrillingRevenuePerfectInfo = DrillingRevenue{Cn,Tech,Tax,PerfectInfo} where {Cn,Tech,Tax}
const DrillingRevenueMaxLearning = DrillingRevenue{Cn,Tech,Tax,MaxLearning} where {Cn,Tech,Tax}

@inline _ρ(x::AbstractLearningType, σ::Number) = _ρ(σ)
@inline _ρ(x::PerfectInfo         , σ::Number) = 1
@inline _ρ(x::MaxLearning         , σ::Number) = 0
@inline _ρ(x::DrillingRevenue     , σ::Number) = _ρ(learn(x), σ)

# @deprecate _ρ(σ::Number, x::AbstractLearningType) _ρ(x,σ)
# @deprecate _ρ(σ::Number, x::PerfectInfo) _ρ(x,σ)
# @deprecate _ρ(σ::Number, x::MaxLearning) _ρ(x,σ)
# @deprecate _ρ(σ::Number, x::DrillingRevenue) _ρ(x,σ)

# parameter access
# -----------------------

@inline _nparm(x::DrillingRevenue{Constrained}) = 2
@inline _nparm(x::DrillingRevenue{Unconstrained, NoTrend}) = 4
@inline _nparm(x::DrillingRevenue{Unconstrained, TimeTrend}) = 5

@inline idx_0(x::DrillingRevenue) = 1
@inline idx_g(x::DrillingRevenue) = 2
@inline idx_ψ(x::DrillingRevenue) = 3
@inline idx_t(x::DrillingRevenueTimeTrend) = 4
@inline idx_ρ(x::DrillingRevenue) = _nparm(x)

@inline theta_0(x::DrillingRevenue, θ) = θ[idx_0(x)]
@inline theta_g(x::DrillingRevenue, θ) = θ[idx_g(x)]
@inline theta_ψ(x::DrillingRevenue, θ) = θ[idx_ψ(x)]
@inline theta_t(x::DrillingRevenue{Unconstrained, TimeTrend}, θ) = θ[idx_t(x)]
@inline theta_ρ(x, θ) = θ[idx_ρ(x)]
function theta_t(x::DrillingRevenue{Unconstrained, NoTrend}, θ)
    # @warn "No t in $x. setting to $STARTING_α_t"
    return STARTING_α_t
end
@inline theta_g(x::DrillingRevenue{Constrained}, θ) = theta_g(constr(x))
@inline theta_ψ(x::DrillingRevenue{Constrained}, θ) = theta_ψ(constr(x))
@inline theta_t(x::DrillingRevenue{Constrained}, θ) = theta_t(constr(x))

function ConstrainedCoefs(x::DrillingRevenueUnconstrained, theta)
    @assert length(theta) == _nparm(x)
    g   = theta_g(x, theta)
    psi = theta_ψ(x, theta)
    t   = theta_t(x, theta)
    out = (log_ogip=g, α_ψ=psi, α_t=STARTING_α_t,)
    return out
end

ConstrainedCoefs(x::DrillReward, θ) = ConstrainedCoefs(revenue(x), vw_revenue(x, θ))

ConstrainedIdx(x::DrillingRevenueTimeTrend) = idx_g(x), idx_ψ(x), idx_t(x)
ConstrainedIdx(x::DrillingRevenueNoTrend) = idx_g(x), idx_ψ(x)

function UnconstrainedFmConstrainedIdx(x::DrillingRevenueConstrained)
    xu = UnconstrainedProblem(x)
    idxu = [idx_0(xu), idx_ρ(xu)]
end

# @deprecate α_0(     x::DrillingRevenue, θ) theta_0(x, θ)
# @deprecate _σ(      x::DrillingRevenue, θ) theta_ρ(x, θ)
# @deprecate log_ogip(x::DrillingRevenue, θ) theta_g(x, θ)
# @deprecate α_ψ(     x::DrillingRevenue, θ) theta_ψ(x, θ)
# @deprecate α_t(     x::DrillingRevenue, θ) theta_t(x, θ)

# taxes
# -----------------------

# From Gulen et al (2015) "Production scenarios for the Haynesville..."
const GATH_COMP_TRTMT_PER_MCF   = 0.42 + 0.07
const MARGINAL_TAX_RATE = 0.402

# other calculations
const REAL_DISCOUNT_AND_DECLINE = 0x1.8e06b611ed4d8p-1  #  0.777394952476835= sum( β^((t+5-1)/12) q(t)/Q(240) for t = 1:240)

const STARTING_α_ψ      = 0x1.7587cc6793516p-2 # 0.365
const STARTING_log_ogip = 0x1.401755c339009p-1 # 0.625
const STARTING_α_t      = 0.01948

struct NoTaxes <: AbstractTaxType end

struct WithTaxes <: AbstractTaxType
    one_minus_mgl_tax_rate::Float64
    cost_per_mcf::Float64
end

struct GathProcess <: AbstractTaxType
    cost_per_mcf::Float64
end

const DrillingRevenueNoTaxes     = DrillingRevenue{Cn,Tech,NoTaxes}     where {Cn,Tech}
const DrillingRevenueWithTaxes   = DrillingRevenue{Cn,Tech,WithTaxes}   where {Cn,Tech}
const DrillingRevenueGathProcess = DrillingRevenue{Cn,Tech,GathProcess} where {Cn,Tech}

WithTaxes(;mgl_tax_rate = 1-MARGINAL_TAX_RATE, cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = WithTaxes(mgl_tax_rate, cost_per_mcf)
GathProcess(;cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = GathProcess(cost_per_mcf)

@inline one_minus_mgl_tax_rate(x::AbstractTaxType) = 1
@inline one_minus_mgl_tax_rate(x::WithTaxes) = x.one_minus_mgl_tax_rate
@inline one_minus_mgl_tax_rate(x) = one_minus_mgl_tax_rate(tax(x))

@inline cost_per_mcf(x::NoTaxes) = 0
@inline cost_per_mcf(x::AbstractTaxType) = x.cost_per_mcf
@inline cost_per_mcf(x) = cost_per_mcf(tax(x))

# royalty
# -----------------------

struct WithRoyalty <: AbstractRoyaltyType end
struct NoRoyalty   <: AbstractRoyaltyType end

@inline (::WithRoyalty)(roy) = roy
@inline (::NoRoyalty)(roy) = 0
@inline oneminusroyalty(x::DrillingRevenue, roy) = 1-royalty(x)(roy)
@inline d_tax_royalty(x::DrillingRevenue, d::Integer, roy::Real) = d * oneminusroyalty(x, roy) * one_minus_mgl_tax_rate(x)

# base functions
# -------------------------------------------

@inline function Eexpψ(x::DrillingRevenueLearn, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    θ4 = theta_ψ(x,θ)

    if _Dgt0(obs)
        return θ4*ψ
    else
        σ = theta_ρ(x,θ)
        ρ = _ρ(learn(x), σ)
        return θ4*(ψ*ρ + θ4*0.5*(1-ρ^2))
    end
end

@inline function dEexpψdα_ψ(x::DrillingRevenueLearn, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    if _Dgt0(obs)
        return ψ
    else
        σ = theta_ρ(x,θ)
        ρ = _ρ(x, σ)
        return ψ*ρ + theta_ψ(x,θ) * (1-ρ^2)
    end
end

@inline function Eexpψ(x::DrillingRevenueNoLearn, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    θ4= theta_ψ(x,θ)
    σ = theta_ρ(x,θ)
    ρ = _ρ(σ)
    return θ4*(ψ*ρ + θ4*0.5*(1-ρ^2))
end

@inline function Eexpψ(x::DrillingRevenueMaxLearning, d, obs, θ, sim)
    θ4= theta_ψ(x,θ)
    if _Dgt0(obs)
        return θ4 * _ψ(obs, sim)
    else
        return 0.5 * θ4^2
    end
end

@inline Eexpψ(x::DrillingRevenuePerfectInfo, d, obs, θ, sim) = theta_ψ(x,θ) * _ψ(obs, sim)

@inline function dEexpψdσ(x::DrillingRevenueLearn, d, obs, θ, sim)
    T = eltype(θ)
    ψ = _ψ(obs, sim)
    if !_Dgt0(obs) && d > 0
        θ4= theta_ψ(x,θ)
        σ = theta_ρ(x,θ)
        out = θ4 *(ψ - θ4*_ρ(σ)) * _dρdθρ(σ)
    else
        out = zero(T)
    end
    return out::T
end


# quantity (for post-estimation SIMULATIONS)
# -----------
function eur_kernel(x::DrillingRevenue, d, obs::ObservationDrill, θ, sim)
    @assert length(θ) == _nparm(x)
    z = zchars(obs)
    logQ = theta_g(x,θ)*geology(obs) + theta_ψ(x,θ)*_ψ2(sim) + trend_component(x, θ, z)
    return exp(logQ)
end

# ----------------------------------------------------------------
# flow revenue
# ----------------------------------------------------------------

function flow(x::DrillingRevenueNoTaxes, d::Integer, obs::ObservationDrill, θ::AbstractVector, sim)
    z = zchars(obs)
    u = d_tax_royalty(x, d, royalty(obs)) *
    exp(
        theta_0(x,θ) + logprice(obs) + theta_g(x,θ)*geology(obs) +
        Eexpψ(x, d, obs, θ, sim) +
        trend_component(x, θ, z)
    )
    return u
end

function flow(x::DrillingRevenue, d::Integer, obs::ObservationDrill, θ::AbstractVector, sim)
    z = zchars(obs)
    u = d_tax_royalty(x, d, royalty(obs)) *
    exp(
        theta_0(x,θ) + theta_g(x,θ)*geology(obs) +
        Eexpψ(x, d, obs, θ, sim) +
        trend_component(x, θ, z)
    ) * (
        price(obs) - cost_per_mcf(x)
    )
    return u
end


# Gradient
# ------------------------------

@inline function flowzero!(grad, θ, dograd)
    dograd && fill!(grad, 0)
    return azero(θ)
end

function flow!(grad, x::DrillingRevenue{Unconstrained, NoTrend}, d, obs, θ, sim, dograd::Bool)
    d == 0 && return flowzero!(grad,θ,dograd)
    rev = flow(x, d, obs, θ, sim)
    if dograd
        grad[idx_0(x)] = rev
        grad[idx_g(x)] = rev*geology(obs)
        grad[idx_ψ(x)] = rev*dEexpψdα_ψ(x, d, obs, θ, sim)
        grad[idx_ρ(x)] = rev*dEexpψdσ(x, d, obs, θ, sim)
    end
    return rev
end

function flow!(grad, x::DrillingRevenue{Unconstrained, TimeTrend}, d, obs, θ, sim, dograd::Bool)
    d == 0 && return flowzero!(grad,θ,dograd)
    z = zchars(obs)
    rev = flow(x, d, obs, θ, sim)
    if dograd
        grad[idx_0(x)] = rev
        grad[idx_g(x)] = rev*geology(obs)
        grad[idx_ψ(x)] = rev*dEexpψdα_ψ(x, d, obs, θ, sim)
        grad[idx_t(x)] = rev*centered_time(x, z)  # FIXME
        grad[idx_ρ(x)] = rev*dEexpψdσ(x, d, obs, θ, sim)
    end
    return rev
end

# Constrained gradient
# ------------------------------

function flow!(grad, x::DrillingRevenue{Constrained}, d, obs, θ, sim, dograd::Bool)
    d == 0 && return flowzero!(grad,θ,dograd)
    rev = flow(x, d, obs, θ, sim)
    if dograd
        grad[idx_0(x)] = rev
        grad[idx_ρ(x)] = rev*dEexpψdσ(x, d, obs, θ, sim)
    end
    return rev
end

coefnames(x::DrillingRevenue{Unconstrained, NoTrend}) =
    ["\\alpha_0", "\\alpha_g", "\\alpha_\\psi", "\\theta_\\rho"]
coefnames(x::DrillingRevenue{Unconstrained, TimeTrend}) =
    ["\\alpha_0", "\\alpha_g", "\\alpha_\\psi", "\\alpha_t", "\\theta_\\rho"]
coefnames(x::DrillingRevenue{Constrained}) =
    ["\\alpha_0", "\\theta_\\rho"]

# ----------------------------------------------------------------
# dψ is the same across many functions
# ----------------------------------------------------------------

function flowdψ(rev::Real, x::DrillingRevenue, d, obs, θ, sim)
    dψ = rev * theta_ψ(x,θ)
    if _Dgt0(obs)
        return dψ
    else
        return dψ * _ρ(x,theta_ρ(x,θ))
    end
end

function flowdψ(x::DrillingRevenue, d, obs, θ, sim)
    d == 0 && return azero(θ)
    rev = flow(x, d, obs, θ, sim)
    return flowdψ(rev, x, d, obs, θ, sim)
end



# Conversion between model types
# ----------------------------------------------------------------

# Constrained / Unconstrained
ConstrainedProblem(  x::AbstractPayoffComponent; kwargs...) = x
UnconstrainedProblem(x::AbstractPayoffComponent; kwargs...) = x

UnconstrainedProblem(x::DrillingRevenue; kwargs...) = DrillingRevenue(Unconstrained(;kwargs...), tech(x), tax(x), learn(x), royalty(x))
ConstrainedProblem(  x::DrillingRevenue; kwargs...) = DrillingRevenue(Constrained(;kwargs...),   tech(x), tax(x), learn(x), royalty(x))

ConstrainedProblem(  x::DrillReward; kwargs...) = DrillReward(ConstrainedProblem(revenue(x); kwargs...), ConstrainedProblem(drill(x)), ConstrainedProblem(extend(x)))
UnconstrainedProblem(x::DrillReward; kwargs...) = DrillReward(UnconstrainedProblem(revenue(x); kwargs...), UnconstrainedProblem(drill(x)), UnconstrainedProblem(extend(x)))
ConstrainedProblem(x::DrillReward, theta::AbstractVector) = ConstrainedProblem(x; ConstrainedCoefs(x,theta)...)

# Learning models
NoLearningProblem(x::AbstractPayoffComponent, args...) = x
LearningProblem(  x::AbstractPayoffComponent, args...) = x

NoLearningProblem(x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), NoLearn(),     royalty(x))
LearningProblem(  x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), Learn(),       royalty(x))
PerfectInfo(      x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), PerfectInfo(), royalty(x))

NoLearningProblem(x::DrillReward, args...) = DrillReward(NoLearningProblem(revenue(x), args...), drill(x), extend(x))
LearningProblem(  x::DrillReward, args...) = DrillReward(  LearningProblem(revenue(x), args...), drill(x), extend(x))


# Royalty
NoRoyaltyProblem(  x::AbstractPayoffComponent, args...) = x
WithRoyaltyProblem(x::AbstractPayoffComponent, args...) = x

NoRoyaltyProblem(  x::DrillingRevenue, args...)      = DrillingRevenue(constr(x), tech(x), tax(x), learn(x), NoRoyalty())
WithRoyaltyProblem(x::DrillingRevenue, args...)      = DrillingRevenue(constr(x), tech(x), tax(x), learn(x), WithRoyalty())

NoRoyaltyProblem(  x::DrillReward, args...) = DrillReward(  NoRoyaltyProblem(revenue(x), args...), drill(x), extend(x))
WithRoyaltyProblem(x::DrillReward, args...) = DrillReward(WithRoyaltyProblem(revenue(x), args...), drill(x), extend(x))
