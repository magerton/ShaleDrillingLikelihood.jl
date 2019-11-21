# -------------------------------------------
# Test model
# -------------------------------------------

"Static discrete choice model to test likelihood"
struct TestDrillModel  <: AbstractDrillModel end
struct TestDrillReward <: AbstractStaticPayoff end
struct TestStateSpace  <: AbstractStateSpace end

const TestDrillModelOrRewardOrData = Union{TestDrillModel,TestDrillReward,DataDrill{<:TestDrillModel}}


reward(    m::TestDrillModel) = TestDrillReward()
statespace(m::TestDrillModel) = TestStateSpace()

length(m::Union{TestDrillModel,TestDrillReward}) = _nparm(m)

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
num_choices(  m::TestStateSpace, args...) = 3
actionspace(  m::TestStateSpace, args...) = 0:2
Dgt0(         m::TestStateSpace, state) = state >= 0
next_state(   m::TestStateSpace, state, d::Integer) = state + d
initial_state(m::TestStateSpace) = 1


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
    m, x, z = _model(obs), _x(obs), zchars(obs)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(m,x,s) +
        theta_drill_x(m,theta)*x +
        theta_drill_z(m,theta)*first(z) +
        theta_drill_d(m,theta)
    )
end

function dflow!(::TestDrillReward, grad, d, obs, theta, s)
    m, x, z = _model(obs), _x(obs), zchars(obs)

    grad[idx_drill_ψ(m)] += d*_ψ(m,x,s)
    grad[idx_drill_x(m)] += d*x
    grad[idx_drill_z(m)] += d*first(z)
    grad[idx_drill_d(m)] += d
    grad[idx_drill_ρ(m)] += d*theta_drill_ψ(m,theta)*_dψdθρ(m,x,s)
    return nothing
end

function flowdψ(x::TestDrillReward, d, obs, theta, s)
    T = eltype(theta)
    m, x, z = _model(obs), _x(obs), zchars(obs)
    u = d*theta_drill_ψ(m,theta)
    return u::T
end
