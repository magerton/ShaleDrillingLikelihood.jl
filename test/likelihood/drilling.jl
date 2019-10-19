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

    theta = [-1.0, 2.0, -2.0, 0.5]
    @test 0.5 == theta_drill_ρ(TestDrillModel(), theta)

    data = DataDrill(
        TestDrillModel(), theta;
        minmaxleases=1:10,
        num_i=500, nperinitial=5:15, nper_development=5:15,
        num_zt=50
    )

    sim = SimulationDraws(500, data)
    println("number of periods is $(length(_y(data)))")

    grad = zeros(length(theta))
    hess = zeros(4,4)
    dtv = DrillingTmpVars(data, theta)

    LL1 = simloglik_drill_data!(grad, hess, data, theta, sim, dtv, true)
    LL2 = simloglik_drill_data!(grad, hess, data, theta, sim, dtv, false)
    @test isfinite(LL1)
    @test isfinite(LL2)
    @test LL1 ≈ LL2

    # check that we don't update gradient here
    simloglik_drill_data!(grad, hess, data, theta, sim, dtv, true)
    gradcopy = copy(grad)
    simloglik_drill_data!(grad, hess, data, theta.*2, sim, dtv, false)
    @test all(grad .== gradcopy)

    # check finite difference
    fdgrad = Calculus.gradient(x -> simloglik_drill_data!(grad, hess, data, x, sim, dtv, false), theta)
    fill!(grad, 0)
    simloglik_drill_data!(grad, hess, data, theta, sim, dtv, true)
    @test grad ≈ fdgrad
    @test !(grad ≈ zeros(length(grad)))

    # @show @benchmark simloglik_drill_data!($grad, $data, $theta, $sim, $dtv, false)
    # @show @benchmark simloglik_drill_data!($grad, $data, $theta, $sim, $dtv, true)

    println("initial tests done")

    let data=data, sim=sim, dtv=dtv

        function f(x)
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(zeros(0), zeros(0,0), data, x, sim, dtv, false)
            return -LL
        end

        function fg!(grad, x)
            fill!(grad, 0)
            hess = zeros(length(grad), length(grad))
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(grad, hess, data, x, sim, dtv, true)
            grad .*= -1
            return -LL
        end

        function h!(hess, x)
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(grad, hess, data, x, sim, dtv, true)
            grad .*= -1
            return -LL
        end

        println("getting ready to optmize")
        # odfg  = OnceDifferentiable(f, fg!, fg!, theta)
        tdfgh = TwiceDifferentiable(f, fg!, fg!, h!, theta)
        res = optimize(tdfgh, theta*0.9, BFGS(), Optim.Options(allow_f_increases=true, show_trace=true))
        @show res
        @show res.minimizer, theta
        # res = optimize(tdfgh, theta*0.9, NelderMead(), Optim.Options(allow_f_increases=true, show_trace=true))
        # @show res

    end



end # testset
end # module
