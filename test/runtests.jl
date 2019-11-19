using Revise
using Base.Threads

module ShaleDrillingLikelihood_NEWFlow_Test

using ShaleDrillingLikelihood
using Test
using Calculus

using ShaleDrillingLikelihood: check_coef_length,
    showtypetree

@testset "Flow gradients" begin
    problem = StaticDrillingPayoff(
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
        DrillingCost_TimeFE(2008,2012),
        DrillingCost_TimeFE(2009,2011),
        DrillingCost_constant(),
        DrillingCost_dgt1(),
        # DrillingCost_TimeFE_rigrate(2008,2012),  # requires a different-sized state-space
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
        ExtensionCost_ψ(),
        StaticDrillingPayoff(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()),
        UnconstrainedProblem( StaticDrillingPayoff(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        ConstrainedProblem(   StaticDrillingPayoff(DrillingRevenue(Unconstrained(),NoTrend(),NoTaxes()), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()), ),
        StaticDrillingPayoff(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant()),
    )

end

    # for f in types_to_test
    #     println("Testing fct $f")
    #     let z = (2.5,2010), ψ = 1.0, geoid = 4.5, roy = 0.25, σ = 0.75
    #
    #         n = length(f)
    #         θ0 = rand(n)
    #         fd = zeros(Float64, n)
    #         g = zeros(Float64, n)
    #
    #         for (d,i) in Iterators.product(0:2, 1:3)
    #
    #             # test ∂f/∂θ
    #             Calculus.finite_difference!((thet) -> flow(f, thet, σ, wp, i, d, z, ψ, geoid, roy), θ0, fd, :central)
    #             ShaleDrillingLikelihood.gradient!(f, θ0, g, σ, wp, i, d, z, ψ, geoid, roy)
    #             @test g ≈ fd
    #
    #             # test ∂f/∂ψ
    #             fdpsi = Calculus.derivative((psi) -> flow(f, θ0, σ, wp, i, d, z, psi, geoid, roy), ψ)
    #             gpsi = flowdψ(f, θ0, σ, wp, i, d, z, ψ, geoid, roy)
    #             @test isapprox(fdpsi, gpsi, atol=1e-4)
    #
    #             # test ∂f/∂σ
    #             fdsig = Calculus.derivative((sig) -> flow(f, θ0, sig, wp, i, d, z, ψ, geoid, roy), σ)
    #             gsig = flowdσ(f, θ0, σ, wp, i, d, z, ψ, geoid, roy)
    #             @test isapprox(fdsig, gsig, atol=1e-4)
    #         end
    #     end
    # end
# end


end # Drilling Model Flow Payoffs


println("using $(nthreads()) threads")

# include("sum-functions.jl")
# include("threadutils.jl")
#
# include("data/data-structure.jl")
# include("data/drilling.jl")
# include("data/overall.jl")
#
# include("drilling-model/flow.jl")

# include("likelihood/royalty.jl")
# include("likelihood/production.jl")
# include("likelihood/drilling.jl")
# include("likelihood/overall.jl")
