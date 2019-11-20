"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end

abstract type AbstractModelVariations end

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
_sgnext(obs) = _y(obs) == 0

_d(obs) = _y(obs)
_Dgt0(obs) = true
_z(obs) = (1.0, 2010,)
_ψ(obs) = 0.0
_ψ2(obs) = 0.0
_roy(obs) = last(_ichars(obs))
_geoid(obs) = first(_ichars(obs))
