export TestDrillModel

actionspace(obs::ObservationDrill) = actionspace(_model(obs),_x(obs))
actionspace(m::AbstractDrillModel, state) = actionspace(m)
actionspace(m::TestDrillModel) = 0:2

num_choices(m::TestDrillModel) = 3
num_choices(m::AbstractDrillModel) = throw(error("num_choices not defined for $m"))
num_choices(obs::ObservationDrill) = num_choices(_model(obs))

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

function check_flow_grad(m::AbstractDrillModel, d::Integer, obs::ObservationDrill, theta::AbstractVector)

    u, v = rand(2)

    # set up simulations
    simdraw(x) = SimulationDraw(u, v, theta_drill_ρ(m, x))
    sim = simdraw(theta)

    # compute analytic
    grad = similar(theta)
    dflow!(grad, d, obs, theta, sim)

    # check finite difference for dθ
    f(x) = flow(d, obs, x, simdraw(x))
    grad ≈ Calculus.gradient(f, theta) || return false

    # check finite difference for dψ
    fdpsi(x) = flow(d, obs, theta, SimulationDrawFD(sim, x))
    dflowdψ(d, obs, theta, sim) ≈ Calculus.derivative(fdpsi, 0.0) || return false

    return true
end






# FOR TESTING ONLY
#---------------------------------------------------------------
const TestObs = ObservationDrill{TestDrillModel}

length(     m::TestDrillModel) = 5

idx_drill_ψ(m::TestDrillModel) = 1
idx_drill_x(m::TestDrillModel) = 2
idx_drill_z(m::TestDrillModel) = 3
idx_drill_d(m::TestDrillModel) = 4
idx_drill_ρ(m::TestDrillModel) = 5

full_payoff(             d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw) = flow(d,obs,theta,s)
dfull_payoff(k::Integer, d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw) = dflow(k,d,obs,theta,s)

function flow(d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(m,x,s) +
        theta_drill_x(m,theta)*x +
        theta_drill_z(m,theta)*first(z) +
        theta_drill_d(m,theta)
    )
end

function dflow(k::Integer, d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    # 1 <= k <= length(theta) || throw(BoundsError(theta,k))
    # check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    k == idx_drill_ψ(m) && return T(d*_ψ(m,x,s))
    k == idx_drill_x(m) && return T(d*x)
    k == idx_drill_z(m) && return T(d*first(z))
    k == idx_drill_d(m) && return T(d)
    k == idx_drill_ρ(m) && return T(d*theta_drill_ψ(m,theta)*_dψdθρ(m,x,s))
end

function dflow!(grad::AbstractVector, d, obs, theta, s)
    @fastmath @inbounds @simd for k in OneTo(length(grad))
        grad[k] = dflow(k, d, obs, theta, s)
    end
end

function dflowdψ(d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return T(d*theta_drill_ψ(m,theta))
end

function dflowdθρ(d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw)
    m = _model(obs)
    k = idx_drill_ρ(m)
    return dflow(k, d, obs, theta, s)
end

@deprecate flowdσ(d,obs,theta,psis) dflowdθρ(d,obs,theta,psis)
