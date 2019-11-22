export AbstractPayoffFunction,
    AbstractStaticPayoff,
    AbstractPayoffComponent,
    AbstractDrillingRevenue,
    AbstractDrillingCost,
    AbstractExtensionCost,
    AbstractStaticPayoff,
    DrillReward,
    DynamicDrillingModel,
    flow, dflow!, flowdψ


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

# @deprecate length(f::AbstractPayoffFunction) _nparm(f)

_nparm(m::AbstractDrillModel) = _nparm(reward(m))

# -------------------------------------------
# Generic functions
# -------------------------------------------

NotDefinedError(m) = throw(error("Not defined for $m"))

# navigate through state space
num_choices(m::AbstractStateSpace, i...) = NotDefinedError(m)
actionspace(m::AbstractStateSpace, i...) = NotDefinedError(m)
_Dgt0(      m::AbstractStateSpace, i...) = NotDefinedError(m)

# navigate through model primitives
reward(    m::AbstractDrillModel) = NotDefinedError(m)
statespace(m::AbstractDrillModel) = NotDefinedError(m)

# From model to things about states
@inline actionspace(m::AbstractDrillModel, i...) = actionspace(statespace(m), i...)
@inline num_choices(m::AbstractDrillModel, i...) = num_choices(statespace(m), i...)
@inline _Dgt0(      m::AbstractDrillModel, i...) = _Dgt0(statespace(m), i...)

@inline _ψ(    m::AbstractDrillModel, state, s::SimulationDraw) = _Dgt0(m, state) ? _ψ2(s) : _ψ1(s)
@inline _dψdθρ(m::AbstractDrillModel, state, s::SimulationDraw) = _Dgt0(m, state) ? azero(s) : _dψ1dθρ(s)

# -------------------------------------------
# some functions to look at stuff
# -------------------------------------------

@inline actionspace(obs::ObservationDrill) = actionspace(_model(obs), _x(obs))
@inline statespace( obs::ObservationDrill) = statespace(_model(obs))
@inline reward(     obs::ObservationDrill) = reward(_model(obs))

@inline _sgnext(            obs::ObservationDrill) = _sgnext(statespace(obs), _x(obs), _y(obs))
@inline _sgnext(d::Integer, obs::ObservationDrill) = _sgnext(statespace(obs), _x(obs), d)
@inline _Dgt0(obs::ObservationDrill) = _Dgt0(statespace(obs), _x(obs))
@inline _d(   obs::ObservationDrill) = _y(obs)

@inline _ψ(    obs::ObservationDrill, ψ::Float64) = ψ
@inline _ψ(    obs::ObservationDrill, s::SimulationDraw) = _ψ(_model(obs), _x(obs), s)
@inline _dψdθρ(obs::ObservationDrill, s::SimulationDraw) = _dψdθρ(_model(obs), _x(obs), s)

# -------------------------------------------
# Payoffs...
# -------------------------------------------

@inline flow(        d, obs, theta, s) = flow(  reward(_model(obs)),           d,   obs, theta, s)
@inline dflow!(grad, d, obs, theta, s) = dflow!(reward(_model(obs)), grad,     d,   obs, theta, s)
@inline flowdψ(grad, d, obs, theta, s) = flowdψ(reward(_model(obs)), grad,     d,   obs, theta, s)
@inline flow(           obs, theta, s) = flow(  reward(_model(obs)),       _y(obs), obs, theta, s)
@inline dflow!(grad,    obs, theta, s) = dflow!(reward(_model(obs)), grad, _y(obs), obs, theta, s)
@inline flowdψ(grad,    obs, theta, s) = flowdψ(reward(_model(obs)), grad, _y(obs), obs, theta, s)

# @deprecate dflowdψ(args...) flowdψ(args...)

function dflow(x::AbstractPayoffFunction, d, obs, theta, s)
    length(theta) == _nparm(x) || throw(DimensionMismatch())
    grad = zero(theta)
    dflow!(grad, x, d, obs, theta, s, true)
    return grad
end

function flow(x::AbstractPayoffFunction, d, obs, theta, s)
    grad = Vector{eltype(theta)}(undef, 0)
    return flow!(grad, x, d, obs, theta, s, false)
end

@inline function full_payoff!(grad, d, obs, theta, sim, dograd)
    return flow!(grad, reward(_model(obs)), d, obs, theta, sim, dograd)
end

@inline function full_payoff(d, obs, theta, sim)
    grad = Vector{eltype(theta)}(undef, 0)
    full_payoff!(grad, d, obs, theta, sim, false)
end


# Check Derivatives
#-------------------------------------------------------------

"check gradient of reward function"
function check_flow_grad(m, d, obs, theta, sim)

    # compute analytic
    grad = zero(theta)
    flow!(grad, m, d, obs, theta, sim, true)

    # check finite difference for dθ
    f(x) = flow!(grad, m, d, obs, x, sim, false)
    @test grad ≈ Calculus.gradient(f, theta)

    # check finite difference for dψ
    fdpsi(x) = flow!(grad, m, d, obs, theta, SimulationDrawFD(sim, x), false)
    @test flowdψ(m, d, obs, theta, sim) ≈ Calculus.derivative(fdpsi, 0.0)
end

function check_flow_grad(m, d, obs, theta)
    u, v = randn(2)
    simdraw(x) = SimulationDraw(u, v, theta_drill_ρ(m, x))
    sim = simdraw(theta)
    check_flow_grad(m,d,obs,theta,sim)
end





# @deprecate flow(x::AbstractDrillModel) reward(x)
