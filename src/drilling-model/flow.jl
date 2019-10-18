actionspace(obs::ObservationDrill) = actionspace(_model(obs),_x(obs))
actionspace(m::AbstractDrillModel, state::Integer) = actionspace(m)
actionspace(m::TestDrillModel) = 0:2

num_choices(m::TestDrillModel) = 3
num_choices(m::AbstractDrillModel) = throw(error("num_choices not defined for $m"))
num_choices(obs::ObservationDrill) = num_choices(_model(obs))

Dgt0(m::AbstractDrillModel, state::Integer) = throw(error("Dgt0 not defined for $(m)"))
Dgt0(m::TestDrillModel,     state::Integer) = state > 1

_ψ(    m::AbstractDrillModel, state::Integer, s::SimulationDraw) = Dgt0(m, state) ? _ψ2(s) : _ψ1(s)
_dψdθρ(m::AbstractDrillModel, state::Integer, s::SimulationDraw) = Dgt0(m, state) ? zero(eltype(s)) : _dψ1dρ(s)

next_state(m::TestDrillModel, state::Integer, d::Integer) = state + d
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

# FOR TESTING ONLY
#---------------------------------------------------------------
const TestObs = ObservationDrill{TestDrillModel}

length(     m::TestDrillModel) = 4

idx_drill_ψ(m::TestDrillModel) = 1
idx_drill_x(m::TestDrillModel) = 2
idx_drill_z(m::TestDrillModel) = 3
idx_drill_ρ(m::TestDrillModel) = 4

full_payoff(d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw) = flow(d,obs,theta,s)

function flow(d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(m,x,s) +
        theta_drill_x(m,theta)*x +
        theta_drill_z(m,theta)*first(z)
    )
end

function dflow(k::Integer, d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    1 <= k <= length(theta) || throw(BoundsError(theta,k))
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    k == idx_drill_ψ(m) && return T(d*_ψ(m,x,s))
    k == idx_drill_x(m) && return T(d*x)
    k == idx_drill_z(m) && return T(d*first(z))
    k == idx_drill_ρ(m) && return T(d*_dψdθρ(m,x,s))
end

function dflowdψ(d::Integer, obs::TestObs, theta::AbstractVector{T}, s::SimulationDraw) where {T}
    check_model_dims(d,obs,theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return T(d*_ψ(m,x,s))
end

function dflowdθρ(d::Integer, obs::TestObs, theta::AbstractVector, s::SimulationDraw)
    m = _model(obs)
    k = idx_drill_ρ(m)
    return dflow(k, d, obs, theta, s)
end

@deprecate flowdσ(d,obs,theta,psis) dflowdθρ(d,obs,theta,psis)
