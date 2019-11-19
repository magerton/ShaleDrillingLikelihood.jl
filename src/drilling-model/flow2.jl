export flow, flowdθ, flowdσ, flowdψ,
    STARTING_α_ψ, STARTING_log_ogip, STARTING_α_t,
    AbstractPayoffFunction,
    AbstractPayoffComponent,
    AbstractDrillingCost,
    AbstractDrillingCost_TimeFE,
    DrillingCost_TimeFE,
    DrillingCost_TimeFE_rigrate,
    DrillingCost_constant,
    DrillingCost_dgt1,
    AbstractExtensionCost,
    ExtensionCost_Constant,
    ExtensionCost_Zero,
    ExtensionCost_ψ,
    AbstractStaticPayoffs,
    StaticDrillingPayoff,
    ConstrainedProblem,
    UnconstrainedProblem,
    AbstractModelVariations,
    AbstractTaxType,
    NoTaxes,
    WithTaxes,
    GathProcess,
    AbstractTechChange,
    NoTrend,
    TimeTrend,
    AbstractConstrainedType,
    Unconstrained,
    Constrained,
    DrillingRevenue,
    Learn,
    NoLearn,
    NoLearningProblem


"Abstract type for payoff components"
abstract type AbstractPayoffComponent end

# payoff components
abstract type AbstractDrillingRevenue <: AbstractPayoffComponent end
abstract type AbstractDrillingCost    <: AbstractPayoffComponent end
abstract type AbstractExtensionCost   <: AbstractPayoffComponent end

"Basic Drilling model"
struct DrillModel{R<:AbstractDrillingRevenue,C<:AbstractDrillingCost,E<:AbstractExtensionCost} <: AbstractDrillModel
    revenue::R
    drill::C
    extend::E
end

const AbstractPayoffFunction = Union{AbstractPayoffComponent, DrillModel}

# deprecate these....
const StaticDrillingPayoff = DrillModel
const AbstractStaticPayoffs = AbstractDrillModel

# -----------------------------------------
# components of stuff
# -----------------------------------------

# access components
revenue(x::DrillModel) = x.revenue
drill(  x::DrillModel) = x.drill
extend( x::DrillModel) = x.extend
@deprecate extensioncost(x::DrillModel) extend(x)
@deprecate drillingcost( x::DrillModel) drill(x)

# -----------------------------------------
# lengths
# -----------------------------------------

@inline length( x::DrillModel) = length(revenue(x)) + length(drill(x)) + length(extend(x)) + 1
@inline lengths(x::DrillModel) = (length(revenue(x)), length(drill(x)), length(extend(x)),)
@inline _nparm(x::DrillModel) = length(x)
@deprecate number_of_model_parms(x::DrillModel) _nparm(x)

# -----------------------------------------
# lengths
# -----------------------------------------
# coeficient ranges
@inline idx_revenue(x::DrillModel) = OneTo(length(revenue(x)))
@inline idx_cost(   x::DrillModel) = OneTo(length(drill(x)))  .+  length(revenue(x))
@inline idx_extend( x::DrillModel) = OneTo(length(extend(x))) .+ (length(revenue(x)) + length(drill(x)))

@deprecate idx_revenue(x::DrillModel) idx_revenue(x)
@deprecate idx_cost(   x::DrillModel) idx_cost(x)
@deprecate idx_extend( x::DrillModel) idx_extend(x)

@inline theta_drill_indexes(x::DrillModel) = (idx_revenue(x), idx_cost(x), idx_extend(x),)
@deprecate coef_ranges(x::DrillModel) theta_drill_indexes(x)

@inline check_coef_length(x::DrillModel, θ) = (length(x) == length(θ) || throw(DimensionMismatch()))

theta_revenue(x::DrillModel, theta) = view(theta, idx_revenue(x))
theta_cost(   x::DrillModel, theta) = view(theta, idx_cost(x))
theta_extend( x::DrillModel, theta) = view(theta, idx_extend(x))
theta_sigma(  x::DrillModel, theta) = last(theta)

# flow???(
#     x::AbstractStaticPayoffs, k::Integer,             # which function
#     θ::AbstractVector, σ::T,                          # parms
#     wp::AbstractUnitProblem, i::Integer, d::Integer,  # follows sprime(wp,i,d)
#     z::Tuple, ψ::T, geoid::Real, roy::T                # other states
# )

function gradient!(f::AbstractPayoffFunction, θ, g, args...)
    K = length(f)
    K == length(θ) == length(g) || throw(DimensionMismatch())
    @inbounds for k = OneTo(K)
        g[k] = flowdθ(f, k, θ, args...)
    end
end

@inline function flow(x::DrillModel, θ, σ::T, obs)::T where {T}
    if d == 0
        u = flow(extend(x), theta_extend(x,θ), σ, obs)
    else
        u = flow(revenue(x), theta_revenue(x,θ), σ, obs) +
            flow(drill(x), theta_cost(x,θ), σ,  obs)
    end
    return u
end

@inline function flowdθ(x::DrillModel, k::Integer, θ, σ::T, obs)::T where {T}
    d == 0 && !_sgnext(obs) && return zero(T)

    kr, kc, ke = lengths(x)

    # revenue
    k < 0              && throw(DomainError(k))
    k <= kr            && return flowdθ(revenue(x), k,       theta_revenue(x,θ), σ, obs)
    k <= kr + kc       && return flowdθ(drill(x),   k-kr,    theta_cost(x,θ),    σ, obs)
    k <= kr + kc + ke  && return flowdθ(extend(x),  k-kr-kc, theta_extend(x,θ),  σ, obs)
    throw(DomainError(k))
end

@inline function flowdψ(x::DrillModel, θ, σ::T, obs)::T where {T}
    if d == 0
        return flowdψ(extend(x), theta_extend(x,θ), σ, obs)
    else
        r = flowdψ(revenue(x), theta_revenue(x,θ), σ, obs)
        c = flowdψ(drill(x), theta_drill(x,θ), σ, obs)
        return r+c
    end
end

@inline function flowdσ(x::DrillModel, θ, σ::T, obs)::T where {T}
    if d == 0
        return flowdσ(extend(x), theta_extend(x,θ), σ, obs)
    else
        r = flowdσ(revenue(x), theta_revenue(x,θ), σ, obs)
        c = flowdσ(drill(x), theta_drill(x,θ), σ, obs)
        return r+c
    end
end

# -------------------------------------------
# Extension
# -------------------------------------------

@inline flowdσ(::AbstractExtensionCost, θ, σ, obs) = zero(σ)
@inline flowdψ(::AbstractExtensionCost, θ, σ, obs) = zero(σ)

"No extension cost"
struct ExtensionCost_Zero <: AbstractExtensionCost end
length(::ExtensionCost_Zero) = 0
@inline flow(  ::ExtensionCost_Zero,    θ, σ, obs) = zero(σ)
@inline flowdθ(::ExtensionCost_Zero, k, θ, σ, obs) = nothing

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
length(::ExtensionCost_Constant) = 1
@inline flow(  ::ExtensionCost_Constant,    θ, σ, obs) = _sgnext(obs) ? θ[1]   : zero(σ)
@inline flowdθ(::ExtensionCost_Constant, k, θ, σ, obs) = _sgnext(obs) ? one(σ) : zero(σ)

"Extension cost depends on ψ"
struct ExtensionCost_ψ <: AbstractExtensionCost end
length(::ExtensionCost_ψ) = 2
@inline flow(  ::ExtensionCost_ψ,    θ, σ, obs)= θ[1] + θ[2]*ψ(obs)
@inline flowdθ(::ExtensionCost_ψ, k, θ, σ, obs)= k == 1 ? one(σ) : ψ(obs)
@inline flowdψ(::ExtensionCost_ψ,    θ, σ, obs)= θ[2]

# -------------------------------------------
# Drilling Cost
# -------------------------------------------

@inline flowdσ(::AbstractDrillingCost, θ, σ, obs) = 0.0
@inline flowdψ(::AbstractDrillingCost, θ, σ, obs) = 0.0

"Single drilling cost"
struct DrillingCost_constant <: AbstractDrillingCost end
@inline length(x::DrillingCost_constant) = 1
@inline flow(  u::DrillingCost_constant,    θ, σ, obs) = _d(obs)*θ[1]
@inline flowdθ(u::DrillingCost_constant, k, θ, σ, obs) = _d(obs)

struct DrillingCost_dgt1 <: AbstractDrillingCost end
@inline length(x::DrillingCost_dgt1) = 2
@inline flow(           u::DrillingCost_dgt1,    θ, σ, obs) = d*(d<=1 ? θ[1] : θ[2])
@inline function flowdθ(u::DrillingCost_dgt1, k, θ, σ, obs)
    d = _y(obs)
    k == 1 && return d <= 1 ? d : 0
    k == 2 && return d >  1 ? d : 0
    throw(DomainError(k))
end


"Abstract Type for Costs w Fixed Effects"
abstract type AbstractDrillingCost_TimeFE <: AbstractDrillingCost end
@inline start(    x::AbstractDrillingCost_TimeFE) = x.start
@inline stop(     x::AbstractDrillingCost_TimeFE) = x.stop
@inline startstop(x::AbstractDrillingCost_TimeFE) = start(x), stop(x)
@inline time_idx( x::AbstractDrillingCost_TimeFE, t) = clamp(t, start(x), stop(x)) - start(x) + 1

"Time FE for 2008-2012"
struct DrillingCost_TimeFE <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE) = 2 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE, θ, σ::T, obs)::T where {T}
    z, d = _z(obs), _y(obs)
    d == 1 && return    θ[time_idx(u,last(z))]
    d  > 1 && return d*(θ[time_idx(u,last(z))] + θ[length(u)])
    d  < 1 && return zero(T)
end
@inline function flowdθ(u::DrillingCost_TimeFE, k, θ, σ, obs)
    z, d = _z(obs), _y(obs)
    if 0 < k <= length(u)-1
        return k == time_idx(u,last(z)) ? d : 0
    end
    k <= length(u) && return d <= 1 ? 0 : d
    throw(DomainError(k))
end


"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE_costdiffs) = 4 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_costdiffs, θ, σ, obs)::T where {T}
    d < 1 && return zero(T)
    K = length(u)
    z, d = _z(obs), _y(obs)
    tidx = time_idx(u, last(z))

    if !_Dgt0(obs)
        r = d == 1 ? zero(T) : θ[K-2]
    else
        r = d == 1 ? θ[K-1]  : θ[K]
    end
    return d*( r + θ[tidx] )
end

@inline function flowdθ(u::DrillingCost_TimeFE_costdiffs, k, θ, σ::T, obs)::T where {T}
    K = length(u)
    z, d, roy, Dgt0 = _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
    tidx = time_idx(u, last(z))
    if 0 < k < K-2
        return k == tidx       ? T(d) : zero(T)

    elseif k == K-2
        return !Dgt0 && d >  1 ? T(d) : zero(T)
    elseif k == K-1
        return  Dgt0 && d == 1 ? one(T) : zero(T)
    elseif k == K
        return  Dgt0 && d >  1 ? T(d) : zero(T)
    else
        throw(DomainError(k))
    end
end


"Time FE w rig rates for 2008-2012"
struct DrillingCost_TimeFE_rigrate <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE_rigrate) = 3 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_rigrate, θ, σ::T, obs)::T where {T}
    z, d = _z(obs), _y(obs)
    d == 1 && return     θ[time_idx(u,last(z))] +                  θ[length(u)]*exp(z[2])
    d  > 1 && return d*( θ[time_idx(u,last(z))] + θ[length(u)-1] + θ[length(u)]*exp(z[2]) )
    d  < 1 && return zero(T)
end
@inline function flowdθ(u::DrillingCost_TimeFE_rigrate, k, θ, σ::T, obs)::T where {T}
    z, d = _z(obs), _y(obs)
    K = length(u)
    if 0 < k <= K-2
        return k == time_idx(u,last(z)) ? T(d) : zero(T)
    end
    k == K-1 && return d <= 1 ? zero(T) : T(d)
    k <= K   && return d == 0 ? zero(T) : d*exp(z[2])
    throw(DomainError(k))
end



"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_rig_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE_rig_costdiffs) = 5 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_rig_costdiffs, θ, σ::T, obs)::T where {T}
    z, d = _z(obs), _y(obs)
    d < 1 && return zero(T)
    K = length(u)
    tidx = time_idx(u, last(z))

    if !_Dgt0(obs)
        r = d == 1 ? zero(T) : θ[K-3]
    else
        r = d == 1 ? θ[K-2]  : θ[K-1]
    end
    return d*( r + θ[tidx] + θ[K]*exp(z[2]) )
end

@inline function flowdθ(u::DrillingCost_TimeFE_rig_costdiffs, k::Integer, θ, σ::T, obs)::T where {T}
    z, d = _z(obs), _y(obs)
    K = length(u)
    tidx = time_idx(u, last(z))
    Dgt0 = _Dgt0(obs)

    if 0 < k < K-3
        return k == tidx       ? T(d) : zero(T)
    elseif k == K-3
        return !Dgt0 && d >  1 ? T(d) : zero(T)
    elseif k == K-2
        return  Dgt0 && d == 1 ? one(T) : zero(T)
    elseif k == K-1
        return  Dgt0 && d >  1 ? T(d) : zero(T)
    elseif k == K
        return  d == 0 ? zero(T) : d*exp(z[2])
    else
        throw(DomainError(k))
    end
end


# ----------------------------------------------------------------
# Drilling revenue variations
# ----------------------------------------------------------------

abstract type AbstractModelVariations end
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

learn(  x::DrillingRevenue) = x.learn
constr( x::DrillingRevenue) = x.constr
tech(   x::DrillingRevenue) = x.tech
tax(    x::DrillingRevenue) = x.tax
learn(  x::DrillingRevenue) = x.learn
royalty(x::DrillingRevenue) = x.royalty

learn(  x::DrillModel) = learn(revenue(x))
constr( x::DrillModel) = constr(revenue(x))
tech(   x::DrillModel) = tech(revenue(x))
tax(    x::DrillModel) = tax(revenue(x))
learn(  x::DrillModel) = learn(revenue(x))
royalty(x::DrillModel) = royalty(revenue(x))

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

# Technology
# -----------------

struct NoTrend <: AbstractTechChange  end
struct TimeTrend <: AbstractTechChange
    baseyear::Int
end
TimeTrend() = TimeTrend(TIME_TREND_BASE)
baseyear(x::TimeTrend) = x.baseyear

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

WithTaxes(;mgl_tax_rate = 1-MARGINAL_TAX_RATE, cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = WithTaxes(mgl_tax_rate, cost_per_mcf)
GathProcess(;cost_per_mcf = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE) = GathProcess(cost_per_mcf)

one_minus_mgl_tax_rate(x::AbstractTaxType) = 1
one_minus_mgl_tax_rate(x::WithTaxes)       = x.one_minus_mgl_tax_rate
one_minus_mgl_tax_rate(x) = one_minus_mgl_tax_rate(tax(x))

cost_per_mcf(x::NoTaxes)         = 0
cost_per_mcf(x::AbstractTaxType) = x.cost_per_mcf
cost_per_mcf(x) = cost_per_mcf(tax(x))

# learning
# -----------------------

struct Learn       <: AbstractLearningType end
struct NoLearn     <: AbstractLearningType end
struct PerfectInfo <: AbstractLearningType end
struct MaxLearning <: AbstractLearningType end

@inline _ρ(σ, x::AbstractLearningType) = _ρ(σ)
@inline _ρ(σ, x::PerfectInfo) = 1
@inline _ρ(σ, x::MaxLearning) = 0

@inline _ρ(σ, x::DrillingRevenue) = _ρ(σ, learn(x))
@inline _ρ(σ, x::DrillModel)      = _ρ(σ, revenue(x))

# royalty
# -----------------------

struct WithRoyalty <: AbstractRoyaltyType end
struct NoRoyalty   <: AbstractRoyaltyType end

# Access parameters
# ----------------------------------------------------------------

@inline log_ogip(x::DrillingRevenue{Constrained}, θ) = log_ogip(constr(x))
@inline α_ψ(     x::DrillingRevenue{Constrained}, θ) = α_ψ(     constr(x))
@inline α_t(     x::DrillingRevenue{Constrained}, θ) = α_t(     constr(x))

@inline log_ogip(x::DrillingRevenue{Unconstrained}, θ) = θ[2]
@inline α_ψ(     x::DrillingRevenue{Unconstrained}, θ) = θ[3]
@inline α_t(     x::DrillingRevenue{Unconstrained}, θ) = θ[4]

constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, NoTrend})   = (log_ogip=2, α_ψ=3,)
constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, TimeTrend}) = (log_ogip=2, α_ψ=3, α_t=4)
constrained_parms(x::DrillModel) = constrained_parms(revenue(x))

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

# base functions
# -------------------------------------------

@inline function Eexpψ(x::AbstractLearningType, θ4, σ, ψ, Dgt0)
    if Dgt0
        return θ4*ψ
    else
        ρ = _ρ(σ, x)
        return θ4*(ψ*ρ + θ4*0.5*(1-ρ^2))
    end
end

@inline function Eexpψ(::NoLearn, θ4, σ, ψ, Dgt0)
    ρ = _ρ(σ)
    return θ4*(ψ*ρ + θ4*0.5*(one(θ4)-ρ^2))
end

@inline Eexpψ(::PerfectInfo, θ4, σ, ψ, Dgt0) = θ4*ψ
@inline Eexpψ(::MaxLearning, θ4, σ, ψ, Dgt0) = Dgt0 ? θ4*ψ : 0.5*θ4^2

@inline Eexpψ(x::DrillingRevenue, θ, σ, obs) = Eexpψ(learn(x), θ4, σ, obs)

# ----------------------------------------------------------------
# regular drilling revenue
# ----------------------------------------------------------------

@inline length(x::DrillingRevenue{Constrained}) = 1
@inline length(x::DrillingRevenue{Unconstrained, NoTrend}) = 3
@inline length(x::DrillingRevenue{Unconstrained, TimeTrend}) = 4

# ----------------------------------------------------------------
# flow revenue
# ----------------------------------------------------------------

@inline d_tax_royalty(x::DrillingRevenue{Cnstr,Trnd,Tax,       Lrn, WithRoyalty}, d, roy) where {Cnstr,Trnd,Lrn,Tax} = d*(1-roy)
@inline d_tax_royalty(x::DrillingRevenue{Cnstr,Trnd,Tax,       Lrn, NoRoyalty},   d, roy) where {Cnstr,Trnd,Lrn,Tax} = d
@inline d_tax_royalty(x::DrillingRevenue{Cnstr,Trnd,WithTaxes, Lrn, WithRoyalty}, d, roy) where {Cnstr,Trnd,Lrn    } = d*(1-roy)*one_minus_mgl_tax_rate(x)
@inline d_tax_royalty(x::DrillingRevenue{Cnstr,Trnd,WithTaxes, Lrn, NoRoyalty},   d, roy) where {Cnstr,Trnd,Lrn    } = d*        one_minus_mgl_tax_rate(x)

# quantity (for post-estimation SIMULATIONS)
# -----------
@inline function eur_kernel(x::DrillingRevenue{<:AbstractConstrainedType,NoTrend}, θ, σ, obs)
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
    return exp(
        log_ogip(x,θ)*geoid +
        α_ψ(x,θ)*_ψ2(obs)
    )
end

@inline function eur_kernel(x::DrillingRevenue{<:AbstractConstrainedType,TimeTrend}, θ, σ, obs)
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
    return exp(
        log_ogip(x,θ)*geoid +
        α_ψ(x,θ)*_ψ2(obs) +
        α_t(x,θ)*(
            last(z) - baseyear(tech(x))
        )
    )
end

# revenue
# -----------

@inline function flow(x::DrillingRevenue{Cn,NoTrend,NoTaxes}, θ, σ::T, obs)::T where {T,Cn}
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)

    u = d_tax_royalty(x, d, roy) * exp(
        θ[1] + z[1] + log_ogip(x,θ)*geoid + Eexpψ(x, α_ψ(x,θ), σ, ψ, _Dgt0(obs))
    )
    return u
end

@inline function flow(x::DrillingRevenue{Cn,TimeTrend,NoTaxes}, θ, σ::T, obs)::T where {T,Cn}
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)

    u = d_tax_royalty(x, d, roy) * exp(
        θ[1] + z[1] + log_ogip(x,θ)*geoid +
        Eexpψ(x, α_ψ(x,θ), σ, ψ, Dgt0) +
        α_t(x,θ)*(last(z) - baseyear(tech(x)))
    )
    return u
end

@inline function flow(x::DrillingRevenue{Cn,NoTrend}, θ, σ::T, obs)::T where {T,Cn}
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)

    u = d_tax_royalty(x, d, roy) * exp(
        θ[1] + log_ogip(x,θ)*geoid + Eexpψ(x, α_ψ(x,θ), σ, ψ, Dgt0)
    ) * (
        exp(z[1]) - cost_per_mcf(x)
    )
    return u
end

@inline function flow(x::DrillingRevenue{Cn,TimeTrend}, θ, σ::T, obs)::T where {T,Cn}
    geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)

    u = d_tax_royalty(x, d, roy) * exp(
        θ[1] + log_ogip(x,θ)*geoid +  Eexpψ(x, α_ψ(x,θ), σ, ψ, Dgt0) +
        α_t(x,θ)*(last(z) - baseyear(tech(x)))
    ) * (
        exp(z[1]) - cost_per_mcf(x)
    )
    return u
end


# ----------------------------------------------------------------
# dψ and dσ are the same across many functions
# ----------------------------------------------------------------

@inline function flowdσ(x::DrillingRevenue, θ, σ, obs)
geoid, ψ, z, d, roy, Dgt0 = _geoid(obs), _ψ(obs), _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
    if !_Dgt0(obs) && _d(obs) > 0
        return flow(x, θ, σ, obs) * (ψ*α_ψ(x,θ) - α_ψ(x,θ)^2*_ρ(σ)) * _dρdσ(σ)
    end
    return zero(σ)
end

@inline function flowdψ(x::DrillingRevenue, θ, σ, obs)
    if _d(obs) > 0
        dψ = flow(x, θ, σ, obs) *  α_ψ(x,θ)
        return _Dgt0(obs) ? dψ : dψ * _ρ(σ)
    end
    return zero(σ)
end

# Constrained derivatives
# ------------------------------

@inline function flowdθ(x::DrillingRevenue{Unconstrained, NoTrend}, k, θ, σ, obs)
    rev = flow(x, θ, σ, obs)
    ψ, Dgt0, z = _ψ(obs), _Dgt0(obs), _z(obs)
    k == 1 && return rev
    k == 2 && return rev*geoid
    k == 3 && return rev*( Dgt0 ? ψ : ψ*_ρ(σ) + θ[3]*(1-_ρ2(σ)))
    throw(DomainError(k))
end

@inline function flowdθ(x::DrillingRevenue{Unconstrained, TimeTrend}, k, θ, σ, obs)
    rev = flow(x, θ, σ, obs)
    ψ, Dgt0, z = _ψ(obs), _Dgt0(obs), _z(obs)
    k == 1 && return rev
    k == 2 && return rev*geoid
    k == 3 && return rev*( Dgt0 ? ψ : ψ*_ρ(σ) + θ[3]*(1-_ρ2(σ)))
    k == 4 && return rev*( last(z) - baseyear(tech(x)) )
    throw(DomainError(k))
end

# Constrained derivatives
# ------------------------------

@inline function flowdθ(x::DrillingRevenue{Constrained}, k, θ, σ, obs)
    rev = flow(x, θ, σ, obs)
    k == 1 && return rev
    throw(DomainError(k))
end
