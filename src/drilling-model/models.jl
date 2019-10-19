"Drilling"
struct DrillModel <: AbstractDrillModel end

# generic functions to access coefs
theta_drill(  d::AbstractDrillModel, theta) = theta

theta_drill_ψ(d::AbstractDrillModel, theta) = theta[idx_drill_ψ(d)]
theta_drill_x(d::AbstractDrillModel, theta) = theta[idx_drill_x(d)]
theta_drill_z(d::AbstractDrillModel, theta) = theta[idx_drill_z(d)]
theta_drill_ρ(d::AbstractDrillModel, theta) = theta[idx_drill_ρ(d)]
theta_drill_d(d::AbstractDrillModel, theta) = theta[idx_drill_d(d)]

"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end
