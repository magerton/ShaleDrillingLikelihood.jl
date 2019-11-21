export AbstractPayoffFunction,
    AbstractStaticPayoff,
    AbstractPayoffComponent,
    AbstractDrillingRevenue,
    AbstractDrillingCost,
    AbstractExtensionCost,
    AbstractStaticPayoff

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

# -------------------------------------------
# Test model
# -------------------------------------------

"Static discrete choice model to test likelihood"
struct TestDrillModel <: AbstractDrillModel end
struct TestDrillReward <: AbstractStaticPayoff end

reward(::TestDrillModel) = TestDrillReward()
@deprecate flow(x::AbstractDrillModel) reward(x)

# generic functions to access coefs
_nparm(d::AbstractDrillModel) = nparm(reward(d))
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
