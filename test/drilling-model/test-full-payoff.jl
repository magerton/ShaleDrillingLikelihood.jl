module ShaleDrillingLikelihood_DynamicDrillingModelTest

DOBTIME = false
PRINTSTUFF = false
DOPROFILE = false

using ShaleDrillingLikelihood
using Test
using SparseArrays
using BenchmarkTools
using Calculus
using Random
using Profile
# using ProfileView
using InteractiveUtils

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: DCDPEmax,
    DCDPTmpVars,
    anticipate_t1ev,
    _dmax,
    solve_vf_all!,
    update_static_payoffs!,
    NoValueFunction,
    ValueFunction,
    ValueFunctionArrayOnly,
    value_function,
    EV,
    dEV,
    ObservationDrill,
    check_flow_grad,
    SimulationDraw,
    idx_ρ,
    actionspace,
    discounted_dynamic_payoff!,
    full_payoff!,
    update_interpolation!

println("print to keep from blowing up")

@testset "Dynamic Drilling Model" begin

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    theta = randn(_nparm(f))
    # @show theta

    nψ, dmx, nz =  13, 3, 5

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-3.0; stop=3.0, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (2.0, 0.25,)

    # ddm object
    ddm_no_t1ev   = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, true)


    @testset "VF Structs" begin
        vfao = ValueFunctionArrayOnly(ddm_no_t1ev)
        vf = ValueFunction(ddm_no_t1ev)
        @test EV(vfao) == EV(vf)
        @test dEV(vfao) == dEV(vf)
    end

    tmpv = DCDPTmpVars(ddm_no_t1ev)

    @testset "VF interpolation" begin


        tmpgrad = similar(theta)
        fdgrad = similar(theta)

        for ddm in (ddm_with_t1ev, ddm_no_t1ev)

            θρ = theta[idx_ρ(reward(ddm))]
            solve_vf_all!(tmpv, ddm, theta, ichar, true)
            update_interpolation!(value_function(ddm), true)

            wp = statespace(ddm)
            for i in OneTo(length(wp))
                for d in actionspace(wp, i)
                    z = clamp(randn(), minimum(zs[1]), maximum(zs[1]))
                    zchars = (z,)
                    obs = ObservationDrill(ddm, ichar, zchars, d, i)
                    uv = randn(2)
                    sim = SimulationDraw(uv..., θρ)

                    check_flow_grad(reward(ddm), d, obs, theta)
                    discounted_dynamic_payoff!(tmpgrad, d, obs, sim, false)
                    discounted_dynamic_payoff!(tmpgrad, d, obs, sim, true)
                    full_payoff!(tmpgrad, d, obs, theta, sim, false)
                    full_payoff!(tmpgrad, d, obs, theta, sim, true)
                end
            end
        end
    end

end



end # module
