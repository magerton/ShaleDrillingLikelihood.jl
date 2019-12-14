module ShaleDrillingLikelihood_DynamicDrillModel_Test_LearningVersions

DOBTIME = false
PRINTSTUFF = false
DOPROFILE = false

using ShaleDrillingLikelihood
using Test
using SparseArrays
using BenchmarkTools
using Random
# using ProfileView
using InteractiveUtils

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: DCDPEmax,
    DCDPTmpVars,
    solve_vf_all_timing!,
    EV,
    solve_vf_all!,
    PerfectInfo,
    MaxLearning,
    _nSexp,
    _ρ,
    end_lrn,
    end_ex0,
    ObservationDrill,
    SimulationDraw,
    flow!,
    vw_revenue,
    learn,
    revenue,
    _Dgt0,
    Eexpψ


println("print to keep from blowing up")

@testset "Versions of Dynamic Drilling Models" begin

    Random.seed!(1234)

    nψ, dmx, nz =  13, 3, 11

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-1.5; stop=1.5, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (2.0, 0.25,)

    @testset "Perfect Info" begin
        f1 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), Learn(),       WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
        f2 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), PerfectInfo(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())

        theta1 = randn(_nparm(f1))
        theta2 = copy(theta1)
        theta1[end] = 20
        theta2[end] = 0

        ddm1 = DynamicDrillModel(f1, 0.9, wp, zs, ztrans, ψs, false)
        ddm2 = DynamicDrillModel(f2, 0.9, wp, zs, ztrans, ψs, false)

        ev1 = ValueFunctionArrayOnly(ddm1)
        ev2 = ValueFunctionArrayOnly(ddm2)
        tmpv = DCDPTmpVars(ddm1)

        fill!(ev1, 0)
        fill!(ev2, 0)

        solve_vf_all!(tmpv, ddm1, theta1, ichar, false)
        solve_vf_all!(tmpv, ddm2, theta2, ichar, false)

        @test EV(ev1) ≈ EV(ev2)
        @test EV(ev1) != EV(ev2)
    end # perfect info




    @testset "Max Learning" begin
        f1 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), Learn(),       WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
        f2 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), MaxLearning(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())

        theta1 = randn(_nparm(f1))
        theta2 = copy(theta1)
        theta1[end] = -20
        theta2[end] = 0

        ddm1 = DynamicDrillModel(f1, 0.9, wp, zs, ztrans, ψs, false)
        ddm2 = DynamicDrillModel(f2, 0.9, wp, zs, ztrans, ψs, false)

        ev1 = ValueFunctionArrayOnly(ddm1)
        ev2 = ValueFunctionArrayOnly(ddm2)
        tmpv = DCDPTmpVars(ddm1)

        fill!(ev1, 0)
        fill!(ev2, 0)

        solve_vf_all!(tmpv, ddm1, theta1, ichar, false)
        solve_vf_all!(tmpv, ddm2, theta2, ichar, false)

        @test EV(ev1) ≈ EV(ev2)
        @test EV(ev1) != EV(ev2)
    end # max learning

    @testset "NoRoyalty" begin
        f1 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), Learn(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
        f2 = DrillReward(DrillingRevenue(Constrained(), NoTrend(), NoTaxes(), Learn(), NoRoyalty()),   DrillingCost_constant(), ExtensionCost_Constant())

        theta1 = randn(_nparm(f1))
        theta2 = copy(theta1)

        ddm1 = DynamicDrillModel(f1, 0.9, wp, zs, ztrans, ψs, false)
        ddm2 = DynamicDrillModel(f2, 0.9, wp, zs, ztrans, ψs, false)

        ev1 = ValueFunctionArrayOnly(ddm1)
        ev2 = ValueFunctionArrayOnly(ddm2)
        tmpv = DCDPTmpVars(ddm1)

        fill!(ev1, 0)
        fill!(ev2, 0)

        solve_vf_all!(tmpv, ddm1, theta1, (2.0, 0.0), false)
        solve_vf_all!(tmpv, ddm2, theta2, (2.0, 0.25), false)

        @test EV(ev1) == EV(ev2)
    end


    @testset "NoLearn" begin

        α0 = -3.0
        αψ0 = 0.35
        sig0 = 0.7
        theta0 = [-5.45746, -0.9, α0, sig0] # α₀, α_cost, α_extend
        ρ0 = _ρ(sig0)
        @test ρ0 < sig0

        α1 = α0 + 0.5*αψ0^2*(1-ρ0^2)
        αψ1 = αψ0*ρ0
        sig1 = 60.0
        theta1 = vcat(theta0[1:2], α1, sig1)

        f0 = DrillReward(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=αψ0, α_t=0.025), NoTrend(), NoTaxes(), NoLearn(),     WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
        f1 = DrillReward(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=αψ1, α_t=0.025), NoTrend(), NoTaxes(), PerfectInfo(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())

        @test learn(revenue(f0)) == NoLearn()

        @test vw_revenue(f0, theta0) == theta0[3:4]
        @test vw_revenue(f1, theta1) == theta1[3:4]

        ddm0 = DynamicDrillModel(f0, 0.9, wp, zs, ztrans, ψs, false)
        ddm1 = DynamicDrillModel(f1, 0.9, wp, zs, ztrans, ψs, false)

        let z = (2.5, 2.0, 2010),
            grad = similar(theta0),
            d = 1,
            i = length(statespace(ddm0))-1,
            obs0 = ObservationDrill(ddm0, ichar, z, d, i),
            obs1 = ObservationDrill(ddm1, ichar, z, d, i),
            u = randn()
            @test _Dgt0(obs0)
            @test _Dgt0(obs1)

            # @show @which Eexpψ(revenue(f0), d, obs0, theta0, u)
            # @show @which Eexpψ(revenue(f1), d, obs1, theta1, u)

            ff0 = flow!(grad, f0, d, obs0, theta0, u, false)
            ff1 = flow!(grad, f1, d, obs1, theta1, u, false)

            @test ff0 ≈ ff1
        end
        ev0 = ValueFunctionArrayOnly(ddm0)
        ev1 = ValueFunctionArrayOnly(ddm1)
        tmpv = DCDPTmpVars(ddm0)

        fill!(ev0, 0)
        fill!(ev1,0)

        solve_vf_all!(tmpv, ddm0, theta0, ichar, false)
        @test all(isfinite.(EV(ev0)))

        solve_vf_all!(tmpv, ddm1, theta1, ichar, false)
        @test all(isfinite.(EV(ev1)))

        @test EV(ev0) ≈ EV(ev1)
    end




end # testset
end # module
