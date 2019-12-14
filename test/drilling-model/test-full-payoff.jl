module ShaleDrillingLikelihood_DynamicDrillModel_ComputeFullPayoff

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
    # @show theta

    nψ, dmx, nz =  13, 3, 5

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-4.5; stop=4.5, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (4.0, 0.25,)

    # ddm object
    ddm_no_t1ev   = DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev = DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, true)


    @testset "VF Structs" begin
        vfao = ValueFunctionArrayOnly(ddm_no_t1ev)
        vf = ValueFunction(ddm_no_t1ev)
        @test EV(vfao) == EV(vf)
        @test dEV(vfao) == dEV(vf)
    end

    tmpv = DCDPTmpVars(ddm_no_t1ev)

    @testset "VF interpolation" begin

        num_parms = _nparm(f)
        gradtmp = zeros(num_parms)
        fdgrad = similar(gradtmp)

        function solve_vf_and_fullpayoff!(gradtmp, tmpv, ddm, d, obs, theta, uv, ichars, dograd=false)
            θρ = theta[idx_ρ(reward(ddm))]
            sim = SimulationDraw(uv..., θρ)
            # fill!(value_function(ddm), 0)
            solve_vf_all!(tmpv, ddm, theta, ichars, dograd)
            update_interpolation!(value_function(ddm), dograd)
            return full_payoff!(gradtmp, d, obs, theta, sim, dograd)
        end

        for ddm in (ddm_with_t1ev, ddm_no_t1ev)
            wp = statespace(ddm)
            for i in OneTo(length(wp))
                for d in actionspace(wp, i)

                    theta = randn(num_parms)
                    z = clamp(randn(), minimum(zs[1]), maximum(zs[1]))
                    obs = ObservationDrill(ddm, ichar, (z,), d, i)
                    uv = randn(2)

                    check_flow_grad(reward(ddm), d, obs, theta)

                    fill!(gradtmp, 0)
                    f(θ) = solve_vf_and_fullpayoff!(gradtmp, tmpv, ddm, d, obs, θ, uv, ichar, false)
                    fd = Calculus.gradient(f, theta)
                    @test all(gradtmp .== 0)

                    solve_vf_and_fullpayoff!(gradtmp, tmpv, ddm, d, obs, theta, uv, ichar, true)
                    @test all(isfinite.(gradtmp))
                    d > 0 && @test any(gradtmp .!= 0)
                    # fd ≈ gradtmp || @show fd .- gradtmp, i, d, z  # prefilter???
                    @test fd ≈ gradtmp
                end
            end
        end
    end

end



end # module
