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
    dflowdθρ,
    SimulationDraw,
    idx_drill_ψ, theta_drill_ψ,
    idx_drill_x, theta_drill_x,
    idx_drill_z, theta_drill_z,
    idx_drill_ρ, theta_drill_ρ,
    TestObs,
    ObservationDrill,
    check_flow_grad

@testset "Drilling Model Flow Payoffs" begin

    @testset "TestDrillModel" begin
        theta = [1.5, 2.0, -3.0, 0.5]
        model = TestDrillModel()
        ichars = [tuple(x) for x in -1.0:0.5:1.0]
        zchars = [tuple(z) for z in -1.0:0.5:1.0]
        y = collect(0:2)
        x = collect(1:4)

        obstypes = product(0:2, product(ichars, zchars, y, x))

        for (d, obstype) in obstypes
            obs = ObservationDrill(model, obstype...)
            @test check_flow_grad(model, d, obs, theta)
        end
    end



end # Drilling Model Flow Payoffs

end # module
