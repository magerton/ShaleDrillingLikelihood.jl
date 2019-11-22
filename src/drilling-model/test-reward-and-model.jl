"Static discrete choice model to test likelihood"
struct TestDrillModel  <: AbstractDrillModel end
struct TestDrillReward <: AbstractStaticPayoff end
struct TestStateSpace  <: AbstractStateSpace end

const TestDrillModelOrRewardOrData = Union{TestDrillModel,TestDrillReward,DataDrill{<:TestDrillModel}}


reward(    m::TestDrillModel) = TestDrillReward()
statespace(m::TestDrillModel) = TestStateSpace()

_nparm(     m::TestDrillModelOrRewardOrData) = 5

idx_drill_ψ(m::TestDrillModelOrRewardOrData) = 1
idx_drill_x(m::TestDrillModelOrRewardOrData) = 2
idx_drill_z(m::TestDrillModelOrRewardOrData) = 3
idx_drill_d(m::TestDrillModelOrRewardOrData) = 4
idx_drill_ρ(m::TestDrillModelOrRewardOrData) = 5

theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d, theta) = theta[idx_drill_x(d)]
theta_drill_z(d, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d, theta) = theta[idx_drill_d(d)]

# Number of parameters
num_choices(  m::TestStateSpace, i...) = 3
actionspace(  m::TestStateSpace, i...) = 0:2
_Dgt0(        m::TestStateSpace, i) = i >= 0
next_state(   m::TestStateSpace, i, d) = state + d
initial_state(m::TestStateSpace) = 1
_sgnext(      m::TestStateSpace, i) = i > 1
_sgnext(      m::TestStateSpace, i, d) = _sgnext(m,i) && d == 0


function check_model_dims(d, obs::ObservationDrill{TestDrillModel}, theta)
    model = _model(obs)
    _nparm(model) == length(theta) || throw(DimensionMismatch())
    d in actionspace(statespace(model), _x(obs)) || throw(BoundsError())
end

# -------------------------------------------
# Test model
# -------------------------------------------

function flow(m::TestDrillReward, d, obs, theta, s)
    check_model_dims(d,obs,theta)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(obs,s) +
        theta_drill_x(m,theta)*_x(obs) +
        theta_drill_z(m,theta)*first(zchars(obs)) +
        theta_drill_d(m,theta)
    )
end

function dflow!(m::TestDrillReward, grad, d, obs, theta, s)
    grad[idx_drill_ψ(m)] = d*_ψ(obs,s)
    grad[idx_drill_x(m)] = d*_x(obs)
    grad[idx_drill_z(m)] = d*first(zchars(obs))
    grad[idx_drill_d(m)] = d
    grad[idx_drill_ρ(m)] = d*theta_drill_ψ(m,theta)*_dψdθρ(obs,s)
    return flow(m, d, obs, theta, s)
end

function flowdψ(m::TestDrillReward, d, obs, theta, s)
    T = eltype(theta)
    u = d*theta_drill_ψ(m,theta)
    return u::T
end
