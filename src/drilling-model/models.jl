"Drilling"
struct DrillModel <: AbstractDrillModel end

"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end

# generic functions to access coefs
theta_drill(  d, theta) = view(theta, 1:length(d))

theta_drill_ψ(d, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d, theta) = theta[idx_drill_x(d)]
theta_drill_z(d, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d, theta) = theta[idx_drill_d(d)]
