module ShaleDrillingLikelihood_Drilling_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads
# using InteractiveUtils

using Calculus
using Optim
using LinearAlgebra

using ShaleDrillingLikelihood: SimulationDraws,
    DataDrill,
    TestDrillModel,
    DrillingTmpVars,
    DrillUnit,
    AbstractDrillRegime,
    DrillLease,
    ObservationDrill,
    InitialDrilling,
    DevelopmentDrilling,
    theta_drill_ρ,
    _model,
    update!,
    loglik_drill_lease!,
    loglik_drill_unit!,
    simloglik_drill_unit!,
    simloglik_drill_data!,
    _y


println("testing drilling likelihood")

@testset "Drilling Likelihood" begin

    Random.seed!(1234)

    theta = [1.0, 1.0, 1.0, 0.5]

    data = DataDrill(
        TestDrillModel(), theta;
        minmaxleases=1:10,
        num_i=100, nperinitial=1:10, nper_development=1:10,
        num_zt=40
    )

    sim = SimulationDraws(500, data)
    println("number of periods is $(length(_y(data)))")

    grad = zeros(length(theta))
    dtv = DrillingTmpVars(data, theta)

    LL1 = simloglik_drill_data!(grad, data, theta, sim, dtv, true)
    LL2 = simloglik_drill_data!(grad, data, theta, sim, dtv, false)
    @test isfinite(LL1)
    @test isfinite(LL2)
    @test LL1 ≈ LL2

    # check that we don't update gradient here
    simloglik_drill_data!(grad, data, theta, sim, dtv, true)
    gradcopy = copy(grad)
    simloglik_drill_data!(grad, data, theta.*2, sim, dtv, false)
    @test all(grad .== gradcopy)

    # check finite difference
    fdgrad = Calculus.gradient(x -> simloglik_drill_data!(grad, data, x, sim, dtv, false), theta)
    fill!(grad, 0)
    simloglik_drill_data!(grad, data, theta, sim, dtv, true)
    @test grad ≈ fdgrad

    @show @benchmark simloglik_drill_data!($grad, $data, $theta, $sim, $dtv, false)
    @show @benchmark simloglik_drill_data!($grad, $data, $theta, $sim, $dtv, true)


end # testset
end # module
