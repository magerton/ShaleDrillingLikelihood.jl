export TestDrillModel

# Number of parameters
length(m::TestDrillModel) = 5
length(m::AbstractDrillModel) = throw(error("Please define `length` for $m"))

# Max number of choices
num_choices(m::TestDrillModel) = 3
num_choices(m::AbstractDrillModel) = throw(error("Please define `num_choices` for $m"))
num_choices(obs::ObservationDrill) = num_choices(_model(obs))

# action space
actionspace(obs::ObservationDrill) = actionspace(_model(obs),_x(obs))
actionspace(m::AbstractDrillModel, state) = actionspace(m)
actionspace(m::TestDrillModel) = 0:2

Dgt0(m::AbstractDrillModel, state) = throw(error("Dgt0 not defined for $(m)"))
Dgt0(m::TestDrillModel,     state) = state >= 0

_ψ(    m::AbstractDrillModel, state, s::SimulationDraw) = Dgt0(m, state) ? _ψ2(s) : _ψ1(s)
_dψdθρ(m::AbstractDrillModel, state, s::SimulationDraw{T}) where {T} = Dgt0(m, state) ? zero(T) : _dψ1dθρ(s)

next_state(m::TestDrillModel, state, d::Integer) = state + d
initial_state(m::TestDrillModel) = 1

function full_payoff(d::Integer,obs::ObservationDrill,theta::AbstractVector,s::SimulationDraw)
    throw(error("not defined for model $(_model(obs))"))
end
# -------------------------------------------------------

function check_model_dims(d::Integer, obs::ObservationDrill, theta::AbstractVector)
    model = _model(obs)
    length(model) == length(theta) || throw(DimensionMismatch())
    d in actionspace(model, _x(obs)) || throw(BoundsError())
end

# Check Derivatives
#-------------------------------------------------------------



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


# FOR TESTING ONLY
#---------------------------------------------------------------
const TestObs = ObservationDrill{TestDrillModel}

idx_drill_ψ(m::Union{TestDrillModel,DataDrill{<:TestDrillModel}}) = 1
idx_drill_x(m::Union{TestDrillModel,DataDrill{<:TestDrillModel}}) = 2
idx_drill_z(m::Union{TestDrillModel,DataDrill{<:TestDrillModel}}) = 3
idx_drill_d(m::Union{TestDrillModel,DataDrill{<:TestDrillModel}}) = 4
idx_drill_ρ(m::Union{TestDrillModel,DataDrill{<:TestDrillModel}}) = 5

full_payoff(             d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw) = flow(d,obs,theta,s)
dfull_payoff(k::Integer, d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw) = dflow(k,d,obs,theta,s)
dfull_payoff!(grad, d, obs::TestObs, theta, sim) = dflow!(grad, d, obs, theta, sim)

function flow(::TestDrillModel, d, obs, theta, s)
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(m,x,s) +
        theta_drill_x(m,theta)*x +
        theta_drill_z(m,theta)*first(z) +
        theta_drill_d(m,theta)
    )
end

flow(d, obs::TestObs, theta, s) = flow(_model(obs), d, obs, theta, s)

function dflow!(::TestDrillModel, grad, d, obs, theta, s)
    m, x, z = _model(obs), _x(obs), zchars(obs)

    grad[idx_drill_ψ(m)] += d*_ψ(m,x,s)
    grad[idx_drill_x(m)] += d*x
    grad[idx_drill_z(m)] += d*first(z)
    grad[idx_drill_d(m)] += d
    grad[idx_drill_ρ(m)] += d*theta_drill_ψ(m,theta)*_dψdθρ(m,x,s)
    return nothing
end

dflow!(grad, d, obs::TestObs, theta, s) = dflow!(_model(obs), grad, d, obs, theta, s)

function dflow(x::TestDrillModel, d, obs, theta, s)
    grad = zero(theta)
    dflow!(x, grad, d, obs, theta, s)
    return grad
end

dflow(d, obs::TestObs, theta, s) = dflow(_model(obs), d, obs, theta, s)

function flowdψ(x::TestDrillModel, d, obs, theta, s)
    T = eltype(theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    u = d*theta_drill_ψ(m,theta)
    return u::T
end

@deprecate dflowdψ(args...) flowdψ(args...)
