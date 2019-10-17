"Drilling"
struct DrillModel <: AbstractDrillModel end


# generic functions to access coefs
theta_drill(  d::AbstractDrillModel, theta) = theta
theta_drill_β(d::AbstractDrillModel, theta) = view(theta, idx_drill_β(d))
theta_drill_ψ(d::AbstractDrillModel, theta) = theta[idx_drill_ψ(d)]
theta_drill_ρ(d::AbstractDrillModel, theta) = theta[idx_drill_ρ(d)]



"Static DrillModel for testing only"
struct TestDrillModel <: AbstractDrillModel end

length(m::TestDrillModel) = 4
actionspace(m::TestDrillModel) = 0:2

idx_drill_β(m::TestDrillModel) = 1:2
idx_drill_ψ(m::TestDrillModel) = 3
idx_drill_ρ(m::TestDrillModel) = 4
