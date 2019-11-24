module ShaleDrillingLikelihood_NEWFlow_Test

using ShaleDrillingLikelihood
using Test
using Calculus

using InteractiveUtils

using Base.Iterators: product
using ShaleDrillingLikelihood: showtypetree,
    ObservationDrill,
    Observation,
    SimulationDraw,
    check_flow_grad,
    TestDrillModel,
    _nparm,
    flow!

using Calculus: finite_difference!

const DOPRINT = false

@testset "Flow gradients" begin
    problem = DrillReward(
        DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()),
        DrillingCost_constant(),
        ExtensionCost_Constant()
    )

    DOPRINT && showtypetree(AbstractPayoffFunction)
    @test true == true

    println("")

    payoff_functions_to_test = (
        DrillingCost_constant(),
        DrillingCost_dgt1(),
        DrillingCost_TimeFE(2008,2012),
        DrillingCost_TimeFE(2009,2011),
        DrillingCost_TimeFE_rigrate(2008,2012),
        DrillingRevenue(Constrained(),NoTrend(),NoTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Constrained(),NoTrend(),WithTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Constrained(),TimeTrend(),NoTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Constrained(),TimeTrend(),WithTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Unconstrained(),NoTrend(),WithTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Unconstrained(),TimeTrend(),NoTaxes(), Learn(), WithRoyalty()),
        DrillingRevenue(Unconstrained(),TimeTrend(),WithTaxes(), Learn(), WithRoyalty()),

        DrillingRevenue(Constrained(),NoTrend(),NoTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Constrained(),NoTrend(),WithTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Constrained(),TimeTrend(),NoTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Constrained(),TimeTrend(),WithTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Unconstrained(),NoTrend(),WithTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Unconstrained(),TimeTrend(),NoTaxes(), Learn(), NoRoyalty()),
        DrillingRevenue(Unconstrained(),TimeTrend(),WithTaxes(), Learn(), NoRoyalty()),

        ExtensionCost_Zero(),
        ExtensionCost_Constant(),
        DrillReward(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), # UnconstrainedProblem( StaticDrillingPayoff(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        ConstrainedProblem(   DrillReward(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant()),
    )

    for f in payoff_functions_to_test
        strout = "Testing $f"
        strout = replace(strout, "ShaleDrillingLikelihood." => "")
        DOPRINT && println(strout)
        let z = (2.5, 2.0, 2010), ichars = (4.5, 0.25)

            n = _nparm(f)
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
            # @code_warntype flow!(g, f, d, obs, theta, s, true)

            for (d,i) in product(0:2, 1:3)

                m = TestDrillModel()
                obs = ObservationDrill(m, ichars, z, d, i)

                check_flow_grad(f, d, obs, θ0, sim)
            end

        end
    end
end


end # Drilling Model Flow Payoffs
