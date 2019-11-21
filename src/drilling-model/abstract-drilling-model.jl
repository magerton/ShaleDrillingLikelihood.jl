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
abstract type AbstractStateSpace end
abstract type AbstractUnitProblem <: AbstractStateSpace end

# generic functions to access coefs
idx_drill(d) = OneTo(_nparm(d))
theta_drill(d, theta) = view(theta, idx_drill(d))

@deprecate length(f::AbstractPayoffFunction) _nparm(f)

# -------------------------------------------
# Generic functions
# -------------------------------------------

NotDefinedError(m) = throw(error("Not defined for $m"))

reward(m::AbstractDrillModel) = NotDefinedError(m)
statespace(m::AbstractDrillModel) = NotDefinedError(m)
_nparm(m::AbstractDrillModel) = _nparm(reward(m))

@deprecate flow(x::AbstractDrillModel) reward(x)

actionspace(m::ObservationDrill, state...) = actionspace(statespace(_model(m)), state...)


num_choices(m::AbstractStateSpace) = NotDefinedError(m)
actionspace(m::AbstractStateSpace) = NotDefinedError(m)
num_choices(m::AbstractDrillModel) = num_choices(statespace(m))
actionspace(m::AbstractDrillModel) = actionspace(statespace(m))

num_choices(m::AbstractStateSpace, state...) = NotDefinedError(m)
actionspace(m::AbstractStateSpace, state...) = NotDefinedError(m)
Dgt0(       m::AbstractStateSpace, state...) = NotDefinedError(m)

num_choices(m::AbstractDrillModel, state...) = num_choices(statespace(m), state...)
actionspace(m::AbstractDrillModel, state...) = actionspace(statespace(m), state...)
Dgt0(       m::AbstractDrillModel, state...) = Dgt0(       statespace(m), state...)



_ψ(    m::AbstractDrillModel, state, s::SimulationDraw) = Dgt0(m, state) ? _ψ2(s) : _ψ1(s)
_dψdθρ(m::AbstractDrillModel, state, s::SimulationDraw{T}) where {T} = Dgt0(m, state) ? zero(T) : _dψ1dθρ(s)


flow(        d, obs, theta, s) = flow(  reward(_model(obs)),       d, obs, theta, s)
dflow!(grad, d, obs, theta, s) = dflow!(reward(_model(obs)), grad, d, obs, theta, s)
flowdψ(grad, d, obs, theta, s) = flowdψ(reward(_model(obs)), grad, d, obs, theta, s)
@deprecate dflowdψ(args...) flowdψ(args...)

function dflow(x::AbstractPayoffFunction, d, obs, theta, s)
    length(theta) == _nparm(x) || throw(DimensionMismatch())
    grad = zero(theta)
    dflow!(x, grad, d, obs, theta, s)
    return grad
end

full_payoff(        d, obs, theta, sim) =  flow(       d, obs, theta, sim)
dfull_payoff!(grad, d, obs, theta, sim) = dflow!(grad, d, obs, theta, sim)
dfull_payoff(       d, obs, theta, sim) = dflow(       d, obs, theta, sim)

# Check Derivatives
#-------------------------------------------------------------

# Check Derivatives
#-------------------------------------------------------------

"check gradient of reward function"
function check_flow_grad(m, d, obs, theta, sim)

    # compute analytic
    grad = zero(theta)
    dflow!(m, grad, d, obs, theta, sim)

    # check finite difference for dθ
    f(x) = flow(m, d, obs, x, sim)
    @test grad ≈ Calculus.gradient(f, theta)

    # check finite difference for dψ
    fdpsi(x) = flow(m, d, obs, theta, SimulationDrawFD(sim, x))
    @test flowdψ(m, d, obs, theta, sim) ≈ Calculus.derivative(fdpsi, 0.0)
end

function check_flow_grad(m, d, obs, theta)
    u, v = randn(2)
    simdraw(x) = SimulationDraw(u, v, theta_drill_ρ(m, x))
    sim = simdraw(theta)
    check_flow_grad(m,d,obs,theta,sim)
end
