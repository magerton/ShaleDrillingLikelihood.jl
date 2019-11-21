export AbstractPayoffFunction,
    AbstractStaticPayoff,
    AbstractPayoffComponent,
    AbstractDrillingRevenue,
    AbstractDrillingCost,
    AbstractExtensionCost,
    AbstractStaticPayoff,
    DrillReward,
    DynamicDrillingModel

# -------------------------------------------
# abstract types
# -------------------------------------------

# Static Payoff
abstract type AbstractPayoffFunction end
abstract type AbstractStaticPayoff    <: AbstractPayoffFunction end
abstract type AbstractPayoffComponent <: AbstractPayoffFunction end

# payoff components
abstract type AbstractDrillingRevenue <: AbstractPayoffComponent end
abstract type AbstractDrillingCost    <: AbstractPayoffComponent end
abstract type AbstractExtensionCost   <: AbstractPayoffComponent end

# also needed
abstract type AbstractUnitProblem end

# -------------------------------------------
# Test model
# -------------------------------------------

"Static discrete choice model to test likelihood"
struct TestDrillModel <: AbstractDrillModel end
struct TestDrillReward <: AbstractStaticPayoff end

reward(::TestDrillModel) = TestDrillReward()
@deprecate flow(x::AbstractDrillModel) reward(x)

# generic functions to access coefs
_nparm(d::AbstractDrillModel) = _nparm(reward(d))
idx_drill(d) = OneTo(_nparm(d))
theta_drill(d, theta) = view(theta, idx_drill(d))

theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d, theta) = theta[idx_drill_x(d)]
theta_drill_z(d, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d, theta) = theta[idx_drill_d(d)]


const TestObs = ObservationDrill{TestDrillModel}
const TestDrillModelOrReward = Union{TestDrillModel,TestDrillReward}
const TestDrill = Union{TestDrillModel,TestDrillReward,TestObs,DataDrill{TestDrillModel}}

# -------------------------------------------
# Actual dynamic model
# -------------------------------------------

"Basic Drilling model"
struct DrillReward{R<:AbstractDrillingRevenue,C<:AbstractDrillingCost,E<:AbstractExtensionCost} <: AbstractStaticPayoff
    revenue::R
    drill::C
    extend::E
end

# -------------------------------------------
# some functions to look at stuff
# -------------------------------------------

_sgnext(wp,i) = true
_sgnext(wp, i, d) = true
_sgnext(obs) = _y(obs) == 0

_d(obs) = _y(obs)
_Dgt0(obs) = true
_z(obs) = (1.0, 2010,)
_ψ(obs) = 0.0
_ψ2(obs) = 0.0

# -------------------------------------------
# full blow dynamic model
# -------------------------------------------

"Full-blown Dynamic discrete choice model"
struct DynamicDrillingModel{T<:Real, PF<:DrillReward, AUP<:AbstractUnitProblem, TT<:Tuple, AM<:AbstractMatrix{T}, AR<:StepRangeLen{T}}
    reward::PF            # payoff function
    discount::T           # discount factor
    statespace::AUP       # structure of endogenous choice vars
    zspace::TT            # z-space (tuple)
    ztransition::AM       # transition for z
    psispace::AR          # ψspace = (u, ρu + sqrt(1-ρ²)*v)
    anticipate_t1ev::Bool # do we anticipate the ϵ shocks assoc w/ each choice?

    function DynamicDrillingModel(reward::APF, discount::T, statespace::AUP, zspace::TT, ztransition::AM, psispace::AR, anticipate_t1ev) where {T,N, APF, AUP, TT<:NTuple{N,AbstractRange}, AM, AR}
        nz = checksquare(ztransition)
        npsi = length(psispace)
        nS = length(statespace)
        nd = length(actionspace(statespace))
        ntheta = _nparm(reward)

        0 < discount < 1 || throw(DomainError(discount))
        nz == prod(lengths.(zspace)) || throw(DimensionMismatch("zspace dim != ztransition dim"))

        return DynamicDrillingModel{T,APF,AUP,TT,AM,AR}(reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev)
    end
end

# deprecate this!
const dcdp_primitives = DynamicDrillingModel

const DrillModel = Union{<:DynamicDrillingModel,<:DrillReward}

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
psispace(       x::DynamicDrillingModel) = x.psispace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev

@deprecate flow(x::DynamicDrillingModel)          reward(x)
@deprecate β(x::DynamicDrillingModel)             discount(x)
@deprecate wp(x::DynamicDrillingModel)            statespace(x)
@deprecate _zspace(x::DynamicDrillingModel)       zspace(x)
@deprecate Πz(x::DynamicDrillingModel)            ztransition(x)
@deprecate _ψspace(x::DynamicDrillingModel)       psispace(x)
@deprecate anticipate_e(x::DynamicDrillingModel)  anticipate_t1ev(x)
