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

    ConstrainedProblem,
    UnconstrainedProblem,
    NoLearningProblem



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

constr( x::DrillingRevenue) = x.constr
tech(   x::DrillingRevenue) = x.tech
tax(    x::DrillingRevenue) = x.tax
learn(  x::DrillingRevenue) = x.learn
royalty(x::DrillingRevenue) = x.royalty

constr( x) = constr(revenue(x))
tech(   x) = tech(revenue(x))
tax(    x) = tax(revenue(x))
learn(  x) = learn(revenue(x))
royalty(x) = royalty(revenue(x))

# Technology
# -----------------

struct NoTrend <: AbstractTechChange  end
struct TimeTrend <: AbstractTechChange
    baseyear::Int
end
TimeTrend() = TimeTrend(TIME_TREND_BASE)
baseyear(x::TimeTrend) = x.baseyear

const DrillingRevenueTimeTrend = DrillingRevenue{Cn,TimeTrend} where {Cn}
const DrillingRevenueNoTrend   = DrillingRevenue{Cn,NoTrend} where {Cn}

@inline centered_time(x::DrillingRevenue, z::Tuple) = last(z) - baseyear(tech(x))
@inline trend_component(x::DrillingRevenueTimeTrend, θ, z) = α_t(x,θ) * centered_time(x, z)
@inline trend_component(x::DrillingRevenueNoTrend, θ, z) = 0

# taxes
# -----------------------

struct NoTaxes <: AbstractTaxType end

struct WithTaxes <: AbstractTaxType
    one_minus_mgl_tax_rate::Float64
    cost_per_mcf::Float64
end

struct GathProcess <: AbstractTaxType
    cost_per_mcf::Float64
end

const DrillingRevenueNoTaxes = DrillingRevenue{Cn,Tech,NoTaxes} where {Cn,Tech}
const DrillingRevenueWithTaxes = DrillingRevenue{Cn,Tech,WithTaxes} where {Cn,Tech}
const DrillingRevenueGathProcess = DrillingRevenue{Cn,Tech,GathProcess} where {Cn,Tech}

WithTaxes(;mgl_tax_rate = 1-MARGINAL_TAX_RATE, cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = WithTaxes(mgl_tax_rate, cost_per_mcf)
GathProcess(;cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = GathProcess(cost_per_mcf)

one_minus_mgl_tax_rate(x::AbstractTaxType) = 1
one_minus_mgl_tax_rate(x::WithTaxes) = x.one_minus_mgl_tax_rate
one_minus_mgl_tax_rate(x) = one_minus_mgl_tax_rate(tax(x))

cost_per_mcf(x::NoTaxes)         = 0
cost_per_mcf(x::AbstractTaxType) = x.cost_per_mcf
cost_per_mcf(x) = cost_per_mcf(tax(x))

# royalty
# -----------------------

struct WithRoyalty <: AbstractRoyaltyType end
struct NoRoyalty   <: AbstractRoyaltyType end

(::WithRoyalty)(roy) = roy
(::NoRoyalty)(roy) = 0
oneminusroyalty(x, roy) = 1-royalty(x)(roy)

@inline d_tax_royalty(x::DrillingRevenue, d, roy) = d * oneminusroyalty(x, roy) * one_minus_mgl_tax_rate(x)

# learning
# -----------------------

struct Learn       <: AbstractLearningType end
struct NoLearn     <: AbstractLearningType end
struct PerfectInfo <: AbstractLearningType end
struct MaxLearning <: AbstractLearningType end

const DrillingRevenueLearn = DrillingRevenue{Cn,Tech,Tax,Learn} where {Cn,Tech,Tax}
const DrillingRevenueNoLearn = DrillingRevenue{Cn,Tech,Tax,NoLearn} where {Cn,Tech,Tax}
const DrillingRevenuePerfectInfo = DrillingRevenue{Cn,Tech,Tax,PerfectInfo} where {Cn,Tech,Tax}
const DrillingRevenueMaxLearning = DrillingRevenue{Cn,Tech,Tax,MaxLearning} where {Cn,Tech,Tax}

@inline _ρ(σ, x::AbstractLearningType) = _ρ(σ)
@inline _ρ(σ, x::PerfectInfo) = 1
@inline _ρ(σ, x::MaxLearning) = 0

@inline _ρ(σ, x::DrillingRevenue) = _ρ(σ, learn(x))
@inline _ρ(σ, x::DrillModel)      = _ρ(σ, revenue(x))

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
log_ogip(x::Constrained) = x.log_ogip
α_ψ(     x::Constrained) = x.α_ψ
α_t(     x::Constrained) = x.α_t

# Access parameters
# ----------------------------------------------------------------

@inline length(x::DrillingRevenue{Constrained}) = 2
@inline length(x::DrillingRevenue{Unconstrained, NoTrend}) = 4
@inline length(x::DrillingRevenue{Unconstrained, TimeTrend}) = 5

@inline α_0(x::DrillingRevenue, θ) = θ[1]
@inline _σ( x::DrillingRevenue, θ) = θ[end]

@inline log_ogip(x::DrillingRevenue{Unconstrained}, θ) = θ[2]
@inline α_ψ(     x::DrillingRevenue{Unconstrained}, θ) = θ[3]
@inline α_t(     x::DrillingRevenue{Unconstrained, TimeTrend}, θ) = θ[4]

@inline log_ogip(x::DrillingRevenue{Constrained}, θ) = log_ogip(constr(x))
@inline α_ψ(     x::DrillingRevenue{Constrained}, θ) = α_ψ(constr(x))
@inline α_t(     x::DrillingRevenue{Constrained, TimeTrend}, θ) = α_t(constr(x))

constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, NoTrend})   = (log_ogip=2, α_ψ=3,)
constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, TimeTrend}) = (log_ogip=2, α_ψ=3, α_t=4)
constrained_parms(x::DrillModel) = constrained_parms(revenue(x))


# base functions
# -------------------------------------------

_ψ(obs::ObservationDrill, s::SimulationDraw) = _Dgt0(obs) ? _ψ2(s) : _ψ1(s)


@inline function Eexpψ(x::DrillingRevenue, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    θ4 = α_ψ(x,θ)

    if _Dgt0(obs)
        return θ4*ψ
    else
        ρ = _ρ(_σ(x,θ), x)
        return θ4*(ψ*ρ + θ4*0.5*(1-ρ^2))
    end
end

@inline function dEexpψdα_ψ(x::DrillingRevenue, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    if _Dgt0(obs)
        return ψ
    else
        ρ = _ρ(_σ(x,θ), x)
        return ψ*ρ + α_ψ(x,θ) * (1-ρ^2)
    end
end

@inline function Eexpψ(x::DrillingRevenueNoLearn, d, obs, θ, sim)
    ψ = _ψ(obs, sim)
    θ4= α_ψ(x,θ)
    ρ = _ρ(_σ(x,θ))
    return θ4*(ψ*ρ + θ4*0.5*(1-ρ^2))
end

@inline function Eexpψ(x::DrillingRevenueMaxLearning, d, obs, θ, sim)
    θ4= α_ψ(x,θ)
    if _Dgt0(obs)
        return θ4 * _ψ(obs, sim)

    else
        return 0.5 * θ4^2
    end
end

@inline Eexpψ(::DrillingRevenuePerfectInfo, d, obs, θ, sim) = α_ψ(x,θ) * _ψ(sim, obs)

@inline function dEexpψdσ(x::DrillingRevenueLearn, d, obs, θ, sim)
    if !_Dgt0(obs) && d > 0
        θ4= α_ψ(x,θ)
        σ = _σ(x,θ)
        return θ4 *(ψ - θ4*_ρ(σ)) * _dρdσ(σ)
    else
        return azero(θ)
    end
end

# ----------------------------------------------------------------
# flow revenue
# ----------------------------------------------------------------

# revenue
# -----------

@inline function flow(x::DrillingRevenueNoTaxes, d, obs, θ, sim) where {Cn,Trnd}
    z, (geoid, roy,) = _z(obs), _ichars(obs)

    u = d_tax_royalty(x, d, roy) *
    exp(
        θ[1] + z[1] + log_ogip(x,θ)*geoid +
        Eexpψ(x, d, obs, θ, sim) +
        trend_component(x, θ, z)
    )
    return u
end


@inline function flow(x::DrillingRevenue, d, obs, θ, sim)
    z, (geoid, roy,) = _z(obs), _ichars(obs)

    u = d_tax_royalty(x, d, roy) * exp(
        θ[1] + log_ogip(x,θ)*geoid +
        Eexpψ(x, d, obs, θ, sim) +
        trend_component(x, θ, z)
    ) * (
        exp(z[1]) - cost_per_mcf(x)
    )
    return u
end


# Constrained derivatives
# ------------------------------

@inline function dflow!(x::DrillingRevenue{Unconstrained, NoTrend}, grad, d, obs, θ, sim)
    d == 0 && return nothing

    z, (geoid, roy,) = _z(obs), _ichars(obs)
    rev = flow(x, d, obs, θ, sim)

    grad[1] += rev
    grad[2] += rev*geoid
    grad[3] += rev*dEexpψdα_ψ(x, d, obs, θ, sim)
    grad[4] += rev*dEexpψdσ(x, d, obs, θ, sim)
    return nothing
end

@inline function dflow!(x::DrillingRevenue{Unconstrained, TimeTrend}, grad, d, obs, θ, sim)
    d == 0 && return nothing

    z, (geoid, roy,) = _z(obs), _ichars(obs)
    rev = flow(x, d, obs, θ, sim)

    grad[1] += rev
    grad[2] += rev*geoid
    grad[3] += rev*dEexpψdα_ψ(x, d, obs, θ, sim)
    grad[4] += rev*centered_time(x, z)
    grad[5] += rev*dEexpψdσ(x, d, obs, θ, sim)
    return nothing
end

# Constrained derivatives
# ------------------------------

@inline function dflow!(x::DrillingRevenue{Constrained}, grad, d, obs, θ, sim)
    d == 0 && return nothing
    rev = flow(x, d, obs, θ, sim)
    grad[1] += rev
    grad[2] += rev*dEexpψdσ(x, d, obs, θ, sim)
    return nothing
end


# ----------------------------------------------------------------
# dψ and dσ are the same across many functions
# ----------------------------------------------------------------

@inline function flowdψ(x::DrillingRevenue, d, obs, θ, sim)
    d == 0 && return azero(θ)
    dψ = flow(x, d, obs, θ, sim) * α_ψ(x,θ)
    if _Dgt0(obs)
        return dψ
    else
        return dψ * _ρ(_σ(x,θ))
    end
end


# quantity (for post-estimation SIMULATIONS)
# -----------
# @inline function eur_kernel(x::DrillingRevenue{<:AbstractConstrainedType,NoTrend}, θ, σ, obs)
#     geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
#     return exp(
#         log_ogip(x,θ)*geoid +
#         α_ψ(x,θ)*_ψ2(obs)
#     )
# end
#
# @inline function eur_kernel(x::DrillingRevenue{<:AbstractConstrainedType,TimeTrend}, θ, σ, obs)
#     geoid, z, roy, Dgt0 = _geoid(obs), _z(obs), _roy(obs), _Dgt0(obs)
#     ψ = _ψ(sim, obs)
#     return exp(
#         log_ogip(x,θ)*geoid +
#         α_ψ(x,θ)*_ψ2(obs) +
#         α_t(x,θ)*(
#             last(z) - baseyear(tech(x))
#         )
#     )
# end



# Conversion between model types
# ----------------------------------------------------------------

# Constrained / Unconstrained
ConstrainedProblem(  x::AbstractPayoffComponent; kwargs...) = x
UnconstrainedProblem(x::AbstractPayoffComponent; kwargs...) = x

UnconstrainedProblem(x::DrillingRevenue; kwargs...) = DrillingRevenue(Unconstrained(;kwargs...), tech(x), tax(x), learn(x), royalty(x))
ConstrainedProblem(  x::DrillingRevenue; kwargs...) = DrillingRevenue(Constrained(;kwargs...),   tech(x), tax(x), learn(x), royalty(x))

ConstrainedProblem(  x::DrillModel; kwargs...) = DrillModel(ConstrainedProblem(revenue(x); kwargs...), ConstrainedProblem(drill(x)), ConstrainedProblem(extend(x)))
UnconstrainedProblem(x::DrillModel; kwargs...) = DrillModel(UnconstrainedProblem(revenue(x); kwargs...), UnconstrainedProblem(drill(x)), UnconstrainedProblem(extend(x)))


# Learning models
NoLearningProblem(x::AbstractPayoffComponent, args...) = x
LearningProblem(  x::AbstractPayoffComponent, args...) = x

NoLearningProblem(x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), NoLearn(),     royalty(x))
LearningProblem(  x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), Learn(),       royalty(x))
PerfectInfo(      x::DrillingRevenue, args...) = DrillingRevenue(constr(x), tech(x), tax(x), PerfectInfo(), royalty(x))

NoLearningProblem(x::DrillModel, args...) = DrillModel(NoLearningProblem(revenue(x), args...), drill(x), extend(x))
LearningProblem(  x::DrillModel, args...) = DrillModel(  LearningProblem(revenue(x), args...), drill(x), extend(x))


# Royalty
NoRoyaltyProblem(  x::AbstractPayoffComponent, args...) = x
WithRoyaltyProblem(x::AbstractPayoffComponent, args...) = x

NoRoyaltyProblem(  x::DrillingRevenue, args...)      = DrillingRevenue(constr(x), tech(x), tax(x), learn(x), NoRoyalty())
WithRoyaltyProblem(x::DrillingRevenue, args...)      = DrillingRevenue(constr(x), tech(x), tax(x), learn(x), WithRoyalty())

NoRoyaltyProblem(  x::DrillModel, args...) = DrillModel(  NoRoyaltyProblem(revenue(x), args...), drill(x), extend(x))
WithRoyaltyProblem(x::DrillModel, args...) = DrillModel(WithRoyaltyProblem(revenue(x), args...), drill(x), extend(x))
