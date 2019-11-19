"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end

# -------------------------------------------
# Drilling payoff has 3 parts
# -------------------------------------------

# generic functions to access coefs
_nparm(d::AbstractDrillModel) = length(d)
idx_drill(d) = OneTo(_nparm(d))
theta_drill(d, theta) = view(theta, idx_drill(d))

theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d, theta) = theta[idx_drill_x(d)]
theta_drill_z(d, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d, theta) = theta[idx_drill_d(d)]



_sgnext(wp,i) = true
_sgnext(wp, i, d) = true
_y(obs)
_d(obs)
_Dgt0(obs) = true
z = _z(obs)
ψ(obs)
_ψ2(obs)
last(z)
z, d, roy, Dgt0 = _z(obs), _y(obs), _roy(obs), _Dgt0(obs)
ψ = _ψ(obs)
