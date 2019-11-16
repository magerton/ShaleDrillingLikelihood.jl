"Drilling"
struct DrillModel <: AbstractDrillModel end

"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end

# generic functions to access coefs
_nparm(d::AbstractDrillModel) = length(d)
idx_drill(d) = OneTo(_nparm(d))
theta_drill(d, theta) = view(theta, idx_drill(d))

theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d, theta) = theta[idx_drill_x(d)]
theta_drill_z(d, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d, theta) = theta[idx_drill_d(d)]
