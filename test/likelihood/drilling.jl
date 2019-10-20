module ShaleDrillingLikelihood_Drilling_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads
using InteractiveUtils
using Distributions

using Calculus
using Optim
using LinearAlgebra

using LinearAlgebra: checksquare

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
    _y,
    update_theta!


println("testing drilling likelihood")

@testset "Drilling Likelihood" begin

    Random.seed!(911)

    theta = [-0.3, 2.0, -2.0, -0.75, 0.5]
    @test 0.5 == theta_drill_ρ(TestDrillModel(), theta)

    data = DataDrill(
        TestDrillModel(), theta;
        minmaxleases=1:1,
        num_i=1_000, nperinitial=10:40, nper_development=10:40,
        num_zt=200, tstart=1:50
    )

    sim = SimulationDraws(1_000, data)
    println("number of periods is $(length(_y(data)))")

    grad = zeros(length(theta))
    hess = zeros(length(theta),length(theta))
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

    # @testset "warntypes for drilling" begin
    #     unit = data[1]
    #     lease = unit[InitialDrilling()][1]
    #     simi = view(sim, 1)
    #     update_theta!(dtv, theta)
    #     @code_warntype loglik_drill_lease!(  grad, lease, theta, simi[1], dtv[1], true)
    #     @code_warntype loglik_drill_unit!(   grad, data[1], theta, simi[1], dtv[1], true)
    #     @code_warntype simloglik_drill_unit!(grad, unit, theta, simi, dtv, true)
    # end


    # check finite difference
    @testset "check finite difference" begin
        let theta = theta.*0.5
            fdgrad = Calculus.gradient(x -> simloglik_drill_data!(grad, hess, data, x, sim, dtv, false), theta)
            fill!(grad, 0)
            simloglik_drill_data!(grad, hess, data, theta, sim, dtv, true)
            @test grad ≈ fdgrad
            @test !(grad ≈ zeros(length(grad)))
        end
    end

    let k = length(theta), hess = zeros(k,k)
        @show @benchmark simloglik_drill_data!($grad, $hess, $data, $theta, $sim, $dtv, false, false)
        @show @benchmark simloglik_drill_data!($grad, $hess, $data, $theta, $sim, $dtv, true, false)
        @show @benchmark simloglik_drill_data!($grad, $hess, $data, $theta, $sim, $dtv, true, true)
    end

    println("initial tests done")

    @testset "drilling likelihood optimization" begin

        k = length(theta)

        function f(x)
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(zeros(k), zeros(k,k), data, x, sim, dtv, false, false)
            return -LL
        end

        function fg!(grad, x)
            tmphess = zeros(k,k)
            fill!(grad, 0)
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(grad, tmphess, data, x, sim, dtv, true, false)
            grad .*= -1
            return -LL
        end

        function h!(hess, x)
            grad = zeros(k)
            checksquare(hess) == length(x) || throw(DimensionMismatch())
            update!(sim, theta_drill_ρ(_model(data), x))
            LL = simloglik_drill_data!(grad, hess, data, x, sim, dtv, true, true)
            grad .*= -1
            return -LL
        end

        function invH0(x::AbstractVector)
            grad = zeros(k)
            hess = zeros(k,k)
            simloglik_drill_data!(grad, hess, data, x, sim, dtv, true, true)
            return inv(hess)
        end

        println("getting ready to optmize")
        odfg  = OnceDifferentiable(f, fg!, fg!, theta)
        tdfgh = TwiceDifferentiable(f, fg!, fg!, h!, theta)
        res = optimize(tdfgh, theta*0.5, BFGS(;initial_invH = invH0), Optim.Options(allow_f_increases=true, show_trace=true))
        @show res
        @show res.minimizer, theta

        vcovinv = invH0(res.minimizer)
        se = sqrt.(diag(vcovinv))
        tstats = res.minimizer ./ se
        pvals = cdf.(Normal(), -2*abs.(tstats))
        coef_and_se = hcat(theta, res.minimizer, se, tstats, pvals)

        err = theta .- res.minimizer
        waldtest = err'*vcovinv*err
        @test ccdf(Chisq(length(theta)), waldtest) > 0.05
        @show coef_and_se
        # Base.showarray(stdout, coef_and_se)
    end



end # testset
end # module
