module ShaleDrillingLikelihood_NEWFlow_Test

using ShaleDrillingLikelihood
using Test
using Calculus
using Random

using InteractiveUtils

using Base.Iterators: product
using ShaleDrillingLikelihood: showtypetree,
    ObservationDrill,
    Observation,
    SimulationDraw,
    check_flow_grad,
    TestDrillModel,
    _nparm,
    flow!,
    theta_ρ,
    vw_revenue,
    vw_cost,
    vw_extend

using Calculus: finite_difference!

const DOPRINT = false

@testset "Drilling Reward" begin

    @testset "Flow gradients" begin
        problem = DrillReward(
            DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()),
            DrillingCost_constant(),
            ExtensionCost_Constant()
        )

        Random.seed!(1234)

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


    @testset "parameter access" begin

        discount = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)  # real discount rate

        # set up coefs
        θρ =  0.0
        αψ = 0.33
        αg = 0.56

        # set up coefs
        #            θρ  ψ        g    x_r     κ₁   κ₂
        θ_royalty = [θρ, 1.0,    0.1, -0.3,  -0.6, 0.6]
        #                αψ, αg, γx   σ2η, σ2u   # η is iwt, u is iw
        θ_produce = vcat(αψ, αg, 0.2, 0.3, 0.4)
        #            drill  ext    α0  αg  αψ  θρ
        θ_drill_u = [-5.5, -2.0, -2.8, αg, αψ, θρ]
        # θ_drill_u = [-5.8,       -2.8, αg, αψ, θρ]
        θ_drill_c = vcat(θ_drill_u[1:3], θρ)

        # model
        rwrd_c = DrillReward(DrillingRevenue(Constrained(;log_ogip=αg, α_ψ=αψ),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
        rwrd_u = UnconstrainedProblem(rwrd_c)
        rwrd = rwrd_u
        @test _nparm(rwrd) == length(θ_drill_u)

        # test drilling coef
        @test theta_ρ(rwrd, θ_drill_u) == θρ
        @test theta_ρ(revenue(rwrd), vw_revenue(rwrd, θ_drill_u)) == θρ
        @test θ_drill_u[1:1] == vw_cost(rwrd, θ_drill_u)
        @test θ_drill_u[2:2] == vw_extend(rwrd, θ_drill_u)
        @test θ_drill_u[end-3:end] == vw_revenue(rwrd, θ_drill_u)


    end

end

end # Drilling Model Flow Payoffs
