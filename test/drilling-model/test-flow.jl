module ShaleDrillingLikelihood_Flow_Test

# using Revise
using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads
# using Profile
# using ProfileView
using InteractiveUtils
using Base.Iterators

using Calculus

using ShaleDrillingLikelihood: TestDrillModel,
    flow,
    dflow,
    full_payoff,
    dflow!,
    dflowdψ,
    SimulationDraw,
    reward,
    idx_drill_ψ, theta_drill_ψ,
    idx_drill_x, theta_drill_x,
    idx_drill_z, theta_drill_z,
    idx_drill_ρ, theta_drill_ρ,
    idx_drill_d, theta_drill_d,
    ObservationDrill,
    check_flow_grad

@testset "Drilling Model Flow Payoffs" begin

    @testset "TestDrillModel" begin
        model = TestDrillModel()
        theta = [1.5, 2.0, -3.0, -0.6, 0.5]
        @test theta_drill_ψ(model, theta) == 1.5
        @test theta_drill_x(model, theta) == 2.0
        @test theta_drill_z(model, theta) == -3.0
        @test theta_drill_d(model, theta) == -0.6
        @test theta_drill_ρ(model, theta) == 0.5

        ichars = [tuple(x) for x in -1.0:0.5:1.0]
        zchars = [tuple(z) for z in -1.0:0.5:1.0]
        y = collect(0:2)
        x = collect(1:4)

        obstypes = product(0:2, product(ichars, zchars, y, x))

        for (d, obstype) in obstypes
            obs = ObservationDrill(model, obstype...)
            check_flow_grad(reward(model), d, obs, theta)
        end
    end



end # Drilling Model Flow Payoffs

end # module
