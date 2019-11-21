module ShaleDrillingLikelihood_NEWFlow_Test

using ShaleDrillingLikelihood
using Test
using Calculus

using InteractiveUtils

using Base.Iterators: product
using ShaleDrillingLikelihood: check_coef_length,
    showtypetree,
    ObservationDrill,
    Observation,
    SimulationDraw,
    check_flow_grad

using Calculus: finite_difference!

@testset "Flow gradients" begin
    problem = DrillReward(
        DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()),
        DrillingCost_constant(),
        ExtensionCost_Constant()
    )

    let θ = fill(0.25, length(problem)),
        σ = 0.75
        check_coef_length(problem, θ)
    end

    showtypetree(AbstractPayoffFunction)
    @test true == true

    println("")

    types_to_test = (
        DrillingCost_constant(),
        DrillingCost_dgt1(),
        DrillingCost_TimeFE(2008,2012),
        DrillingCost_TimeFE(2009,2011),
        DrillingCost_TimeFE_rigrate(2008,2012),
        DrillingRevenue(Constrained(),NoTrend(),NoTaxes()),
        DrillingRevenue(Constrained(),NoTrend(),WithTaxes()),
        DrillingRevenue(Constrained(),TimeTrend(),NoTaxes()),
        DrillingRevenue(Constrained(),TimeTrend(),WithTaxes()),
        DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()),
        DrillingRevenue(Unconstrained(),NoTrend(),WithTaxes()),
        DrillingRevenue(Unconstrained(),TimeTrend(),NoTaxes()),
        DrillingRevenue(Unconstrained(),TimeTrend(),WithTaxes()),
        ExtensionCost_Zero(),
        ExtensionCost_Constant(),
        DrillReward(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), # UnconstrainedProblem( StaticDrillingPayoff(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        ConstrainedProblem(   DrillReward(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant()),
    )

    for f in types_to_test
        strout = "Testing $f"
        strout = replace(strout, "ShaleDrillingLikelihood." => "")
        println(strout)
        let z = (2.5, 2.0, 2010), ichars = (4.5, 0.25)

            n = length(f)
            θ0 = rand(n)
            u, v = randn(2)

            if n > 0
                sim = SimulationDraw(u, v, last(θ0))
            else
                sim = SimulationDraw(u, v, 0.0)
            end

            # m = TestDrillModel()
            # (d,i) = (1,1)
            # theta = θ0
            # s = sim
            # obs = ObservationDrill(m, ichars, z, d, i)
            # g = zero(theta)
            # @code_warntype flow(f, d, obs, theta, s)
            # @code_warntype dflow!(f, g, d, obs, theta, s)

            for (d,i) in product(0:2, 1:3)

                m = TestDrillModel()
                obs = ObservationDrill(m, ichars, z, d, i)

                check_flow_grad(f, d, obs, θ0, sim)
            end

        end
    end
end


end # Drilling Model Flow Payoffs
