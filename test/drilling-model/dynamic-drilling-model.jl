module ShaleDrillingLikelihood_DynamicDrillingModelTest

using ShaleDrillingLikelihood
using Test
using SparseArrays
using BenchmarkTools
using Calculus
using Random

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: DynamicDrillingModel,
    DCDPEmax,
    DCDPTmpVars,
    _ubVfull    ,
    _dubVfull   ,
    _q          ,
    _lse        ,
    _tmp        ,
    _tmp_cart   ,
    _Πψtmp      ,
    _IminusTEVp,
    flow!,
    end_ex0,
    _βΠψ!,
    _βΠψdθρ!,
    check_dΠψ,
    _dρdθρ,
    _dρdσ,
    theta_ρ,
    reward,
    peturb,
    vw_revenue,
    idx_ρ,
    revenue,
    _dψ1dρ,
    _dψ1dθρ,
    _ψ1,
    _ρ


@testset "Dynamic Drilling Model" begin

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    theta = randn(_nparm(f))
    # @show theta

    nψ, dmx, nz =  13, 3, 11

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-1.5; stop=1.5, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans ./= sum(ztrans; dims=2)

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ddm object
    ddm = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, true)

    evs = DCDPEmax(ddm)
    tmpv = DCDPTmpVars(ddm)

    fd = similar(_dubVfull(tmpv))
    ubvminus = zero(_ubVfull(tmpv))
    ubvplus  = zero(_ubVfull(tmpv))
    thetaminus = similar(theta)
    thetaplus = similar(theta)

    @testset "Fill per-period flow payoffs" begin
        for sidx in 1:length(wp)
            T = eltype(theta)
            for i in OneTo(length(theta))
                thetaminus .= theta
                thetaplus .= theta
                h = peturb(theta[i])
                thetaplus[i] += h
                thetaminus[i] -= h
                twoh = thetaplus[i] - thetaminus[i]
                @test maximum(abs.(thetaplus .- thetaminus)) ≈ twoh

                @views tmpvminus = DCDPTmpVars(ubvminus, _dubVfull(tmpv), _q(tmpv), _lse(tmpv), _tmp(tmpv), _tmp_cart(tmpv), _Πψtmp(tmpv), _IminusTEVp(tmpv))
                @views tmpvplus  = DCDPTmpVars(ubvplus,  _dubVfull(tmpv), _q(tmpv), _lse(tmpv), _tmp(tmpv), _tmp_cart(tmpv), _Πψtmp(tmpv), _IminusTEVp(tmpv))

                fill!(_dubVfull(tmpv), 0)
                fill!(_ubVfull(tmpvminus), 0)
                fill!(_ubVfull(tmpvplus), 0)

                flow!(tmpvminus, ddm, thetaminus, sidx, (2.0, 0.25,), false)
                flow!(tmpvplus,  ddm, thetaplus, sidx, (2.0, 0.25,), false)
                @test all(_dubVfull(tmpv) .== 0)

                fd[i,:,:,:] .= (ubvplus .- ubvminus)./ twoh
            end

            flow!(tmpv, ddm, theta, sidx, (2.0, 0.25,), true)

            fdnosig = fd[1:end-1,:,:,:] .- _dubVfull(tmpv)[1:end-1,:,:,:]
            fdsig = fd[end,:,:,:] .- _dubVfull(tmpv)[end,:,:,:]

            # @views maxv, idx = findmax(abs.(fdnosig))
            # println("worst value is $maxv at $(CartesianIndices(fdnosig)[idx])")
            #
            # @views maxv, idx = findmax(abs.(fdsig))
            # println("worst value is $maxv at $(CartesianIndices(fdsig)[idx])")

            @test fd ≈ _dubVfull(tmpv)
        end

        @btime flow!($tmpv, $ddm, $theta, 2, (2.0, 0.25,), true)
    end

    @testset "Learning transition" begin
        check_dΠψ(ddm, theta)
        _βΠψ!(tmpv, ddm, theta)
        _βΠψdθρ!(tmpv, ddm, theta)


        @testset "learning transition small functions" begin
            for x0 ∈ (0.25, 0.5, 0.75,)
                vspace = range(-2.0, stop=2.0, length=5)
                @test Calculus.derivative(_ρ, x0) ≈ _dρdθρ(x0)

                for u in vspace
                    for v in vspace
                        @test Calculus.derivative((z) -> _ψ1(u,v,_ρ(z)), x0) ≈ _dψ1dθρ(u,v,_ρ(x0), x0)
                        @test Calculus.derivative((z) -> _ψ1(u,v,z), x0) ≈ _dψ1dρ(u,v,x0)
                    end
                end
            end
        end


        @testset "Frechet derivative of ψ transition matrix" begin

            rwrd = reward(ddm)
            σv = theta_ρ(rwrd, theta)
            h = peturb(σv)
            σ1 = σv - h
            σ2 = σv + h
            hh = σ2 - σ1

            theta1 = copy(theta)
            theta2 = copy(theta)
            setindex!(vw_revenue(rwrd, theta1), σ1, idx_ρ(revenue(rwrd)))
            setindex!(vw_revenue(rwrd, theta2), σ2, idx_ρ(revenue(rwrd)))
            hh = theta_ρ(reward(ddm), theta2) - theta_ρ(reward(ddm), theta1)
            @test hh > 0

            pi1 = similar(_Πψtmp(tmpv))
            pi2 = similar(_Πψtmp(tmpv))
            fdpi = similar(_Πψtmp(tmpv))

            tmpv1 = DCDPTmpVars(_ubVfull(tmpv), _dubVfull(tmpv), _q(tmpv), _lse(tmpv), _tmp(tmpv), _tmp_cart(tmpv), pi1, _IminusTEVp(tmpv))
            tmpv2 = DCDPTmpVars(_ubVfull(tmpv), _dubVfull(tmpv), _q(tmpv), _lse(tmpv), _tmp(tmpv), _tmp_cart(tmpv), pi2, _IminusTEVp(tmpv))

            _βΠψ!(tmpv1, ddm, theta1)
            _βΠψ!(tmpv2, ddm, theta2)
            _βΠψdθρ!(tmpv, ddm, theta)
            fdpi .= -pi1
            fdpi .+= pi2
            fdpi ./= hh

            dpi = _Πψtmp(tmpv)

            @views maxv, idx = findmax(abs.(dpi .- fdpi))
            sub = CartesianIndices(fdpi)[idx]
            @show "worst value is $maxv at $sub for β dΠ/dσ"

            @test 0.0 < maxv < 1e-5
            @test fdpi ≈ dpi

        end

    end

end # testset
end # module
