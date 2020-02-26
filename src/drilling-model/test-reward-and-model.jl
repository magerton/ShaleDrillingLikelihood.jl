export TestDrillModel, TestDrillReward

"Static discrete choice model to test likelihood"
struct TestDrillModel  <: AbstractStaticDrillModel end
struct TestDrillReward <: AbstractStaticPayoff end
struct TestStateSpace  <: AbstractStateSpace end

const TestDrillModelOrRewardOrData = Union{TestDrillModel,TestDrillReward,DataDrill{<:TestDrillModel}}


@inline reward(    m::TestDrillModel) = TestDrillReward()
@inline statespace(m::TestDrillModel) = TestStateSpace()

@inline _nparm(     m::TestDrillModelOrRewardOrData) = 5

@inline idx_drill_ψ(m::TestDrillModelOrRewardOrData) = 1
@inline idx_drill_x(m::TestDrillModelOrRewardOrData) = 2
@inline idx_drill_z(m::TestDrillModelOrRewardOrData) = 3
@inline idx_drill_d(m::TestDrillModelOrRewardOrData) = 4
@inline idx_drill_ρ(m::TestDrillModelOrRewardOrData) = 5

@inline theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
@inline theta_drill_x(d, theta) = theta[idx_drill_x(d)]
@inline theta_drill_z(d, theta) = theta[idx_drill_z(d)]
@inline theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
@inline theta_drill_d(d, theta) = theta[idx_drill_d(d)]

coefnames(m::TestDrillReward) = ["\\psi", "x", "z", "d", "\\theta\\rho"]

# Number of parameters
@inline num_choices(  m::TestStateSpace, i...) = 3
@inline actionspace(  m::TestStateSpace, i...) = 0:2
@inline _Dgt0(        m::TestStateSpace, i) = i >= 0
@inline next_state(   m::TestStateSpace, i, d) = state + d
@inline initial_state(m::TestStateSpace) = 1
@inline _sgnext(      m::TestStateSpace, i) = i > 1
@inline _sgnext(      m::TestStateSpace, i, d) = _sgnext(m,i) && d == 0
expires_today(m::TestStateSpace, i) = i == 2
expires_today(m::TestStateSpace, i, d) = expires_today(m,i) && d == 0


function check_model_dims(d, obs::ObservationDrill{TestDrillModel}, theta)
    model = _model(obs)
    _nparm(model) == length(theta) || throw(DimensionMismatch())
    d in actionspace(statespace(model), _x(obs)) || throw(BoundsError())
end

# -------------------------------------------
# Test model
# -------------------------------------------

@inline function flow(m::TestDrillReward, d, obs, theta, s)
    check_model_dims(d,obs,theta)
    return d*(
        theta_drill_ψ(m,theta)*_ψ(obs,s) +
        theta_drill_x(m,theta)*_x(obs) +
        theta_drill_z(m,theta)*first(zchars(obs)) +
        theta_drill_d(m,theta)
    )
end

@inline function flow!(grad, m::TestDrillReward, d, obs, theta, s, dograd::Bool)
    if dograd
        grad[idx_drill_ψ(m)] = d*_ψ(obs,s)
        grad[idx_drill_x(m)] = d*_x(obs)
        grad[idx_drill_z(m)] = d*first(zchars(obs))
        grad[idx_drill_d(m)] = d
        grad[idx_drill_ρ(m)] = d*theta_drill_ψ(m,theta)*_dψdθρ(obs,s)
    end
    return flow(m, d, obs, theta, s)
end

@inline function flowdψ(m::TestDrillReward, d, obs, theta, s)
    T = eltype(theta)
    u = d*theta_drill_ψ(m,theta)
    return u::T
end
