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
    ubVfull    ,
    dubVfull   ,
    q          ,
    lse        ,
    tmp        ,
    tmp_cart   ,
    Πψtmp      ,
    IminusTEVp,
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

    fd = similar(dubVfull(tmpv))
    ubvminus = zero(ubVfull(tmpv))
    ubvplus  = zero(ubVfull(tmpv))
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

                @views tmpvminus = DCDPTmpVars(ubvminus, dubVfull(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv))
                @views tmpvplus  = DCDPTmpVars(ubvplus,  dubVfull(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv))

                fill!(dubVfull(tmpv), 0)
                fill!(ubVfull(tmpvminus), 0)
                fill!(ubVfull(tmpvplus), 0)

                flow!(tmpvminus, ddm, thetaminus, sidx, (2.0, 0.25,), false)
                flow!(tmpvplus,  ddm, thetaplus, sidx, (2.0, 0.25,), false)
                @test all(dubVfull(tmpv) .== 0)

                fd[i,:,:,:] .= (ubvplus .- ubvminus)./ twoh
            end

            flow!(tmpv, ddm, theta, sidx, (2.0, 0.25,), true)

            fdnosig = fd[1:end-1,:,:,:] .- dubVfull(tmpv)[1:end-1,:,:,:]
            fdsig = fd[end,:,:,:] .- dubVfull(tmpv)[end,:,:,:]

            # @views maxv, idx = findmax(abs.(fdnosig))
            # println("worst value is $maxv at $(CartesianIndices(fdnosig)[idx])")
            #
            # @views maxv, idx = findmax(abs.(fdsig))
            # println("worst value is $maxv at $(CartesianIndices(fdsig)[idx])")

            @test fd ≈ dubVfull(tmpv)
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

            pi1  = similar(Πψtmp(tmpv))
            pi2  = similar(Πψtmp(tmpv))
            fdpi = similar(Πψtmp(tmpv))

            tmpv1 = DCDPTmpVars(ubVfull(tmpv), dubVfull(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), pi1, IminusTEVp(tmpv))
            tmpv2 = DCDPTmpVars(ubVfull(tmpv), dubVfull(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), pi2, IminusTEVp(tmpv))

            _βΠψ!(tmpv1, ddm, theta1)
            _βΠψ!(tmpv2, ddm, theta2)
            _βΠψdθρ!(tmpv, ddm, theta)
            fdpi .= -pi1
            fdpi .+= pi2
            fdpi ./= hh

            dpi = Πψtmp(tmpv)

            @views maxv, idx = findmax(abs.(dpi .- fdpi))
            sub = CartesianIndices(fdpi)[idx]
            @show "worst value is $maxv at $sub for β dΠ/dσ"

            @test 0.0 < maxv < 1e-5
            @test fdpi ≈ dpi

        end
    end




    # let EV         = evs.EV,
    #     dEV        = evs.dEV,
    #     σ          = σv,
    #     wp         = prim.wp,
    #     Πz         = prim.Πz,
    #     β          = prim.β,
    #     fdEV       = similar(evs.dEV),
    #     θ1         = similar(θt),
    #     θ2         = similar(θt),
    #     roy        = 0.25,
    #     geoid      = 4.7,
    #     itype      = (geoid, roy,),
    #     nS         = length(prim.wp),
    #     i          = nS-1,
    #     idxd       = ShaleDrillingModel.dp1space(wp,i),
    #     idxs       = collect(ShaleDrillingModel.sprimes(wp,i)),
    #     horzn      = ShaleDrillingModel._horizon(wp,i)
    #
    #     @views ubV =    tmpv.ubVfull[:,:,idxd]
    #     @views dubV =   tmpv.dubVfull[:,:,:,idxd]
    #     @views dubV_σ = tmpv.dubV_σ[:,:,idxd]
    #     @views q      = tmpv.q[:,:,idxd]
    #
    #     @views EV0 = EV[:,:,i]
    #     @views EV1 = EV[:,:,idxs]
    #     @views dEV0 = dEV[:,:,:,i]
    #     @views dEV1 = dEV[:,:,:,idxs]
    #     @views fdEV0 = fdEV[:,:,:,i]
    #
    #
    #     for anticipate_e in (true,false,)
    #
    #         tmpv_vw = dcdp_tmpvars(ubV, dubV, dubV_σ, q, tmpv.lse, tmpv.tmp, tmpv.tmp_cart, tmpv.Πψtmp, tmpv.IminusTEVp)
    #         p = dcdp_primitives(flowfuncname, prim.β, prim.wp, prim.zspace, prim.Πz, prim.ψspace, anticipate_e)
    #
    #         @testset "Check finite horizon gradient for anticipate_e = $anticipate_e" begin
    #             for k = 1:length(θt)
    #                 h = peturb(θt[k])
    #                 θ1 .= θt
    #                 θ2 .= θt
    #                 θ1[k] -= h
    #                 θ2[k] += h
    #                 hh = θ2[k] - θ1[k]
    #
    #                 fill!(EV, 0.0)
    #                 fillflows!(ubV, flow, p, θ1, σ, i, itype...)
    #                 @test !all(ubV .== 0.0)
    #                 ubV .+= β .* EV1
    #                 ShaleDrillingModel.vfit!(EV0, tmpv_vw, p)
    #                 fdEV0[:,:,k] .= -EV0
    #
    #                 fill!(EV, 0.0)
    #                 fillflows!(ubV, flow, p, θ2, σ, i, itype...)
    #                 ubV .+= β .* EV1
    #                 ShaleDrillingModel.vfit!(EV0, tmpv_vw, p)
    #                 fdEV0[:,:,k] .+= EV0
    #                 fdEV0[:,:,k] ./= hh
    #             end
    #
    #             fill!(EV, 0.0)
    #             fill!(dEV, 0.0)
    #             fillflows!(      ubV, flow,   p, θt, σ, i, itype...)
    #             fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
    #             ubV .+= β .* EV1
    #             dubV .+= β .* dEV1
    #             ShaleDrillingModel.vfit!(EV0, dEV0, tmpv_vw, p)
    #
    #             println("extrema(dEV0) = $(extrema(dEV0))")
    #             @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
    #             @views sub = CartesianIndices(fdEV0)[idx]
    #             println("worst value is $maxv at $sub for dθ")
    #
    #             @test 0.0 < maxv < 1.0
    #             @test all(isfinite.(dEV0))
    #             @test all(isfinite.(fdEV0))
    #             @test fdEV0 ≈ dEV0
    #         end
    #
    #         @testset "Check vfit/pfit for infinite horizon, anticipate_e = $anticipate_e" begin
    #             vfEV0 = zeros(size(EV0))
    #
    #             fill!(EV, 0.0)
    #             fillflows!(ubV, flow, p, θt, σ, i, itype...)
    #             ubV .+= β .* EV1
    #             converged, iter, bnds = ShaleDrillingModel.solve_inf_vfit!(EV0, tmpv_vw, p; maxit=1000, vftol=1e-10)
    #             println("vfit done. converged = $converged after $iter iterations. error bounds are $bnds")
    #             vfEV0 .= EV0
    #
    #             fill!(EV, 0.0)
    #             fillflows!(ubV, flow, p, θt, σ, i, itype...)
    #             ubV .+= β .* EV1
    #             converged, iter, bnds = ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, tmpv_vw, p; vftol=1e-10, maxit0=20, maxit1=40)
    #             println("pfit done. converged = $converged after $iter iterations. error bounds are $bnds")
    #
    #             @views maxv, idx = findmax(abs.(vfEV0 .- EV0))
    #             @views sub = CartesianIndices(EV0)[idx]
    #             println("worst value is $maxv at $sub for vfit vs pfit")
    #             @test EV0 ≈ vfEV0
    #         end
    #
    #         @testset "Check gradient for infinite horizon, anticipate_e = $anticipate_e" begin
    #
    #             for k = 1:length(θt)
    #                 h = peturb(θt[k])
    #                 θ1 .= θt
    #                 θ2 .= θt
    #                 θ1[k] -= h
    #                 θ2[k] += h
    #                 hh = θ2[k] - θ1[k]
    #
    #                 fill!(EV, 0.0)
    #                 fillflows!(ubV, flow, p, θ1, σ, i, itype...)
    #                 ubV .+= β .* EV1
    #                 ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, tmpv_vw, p; vftol=1e-10, maxit0=20, maxit1=40)
    #                 fdEV0[:,:,k] .= -EV0
    #
    #                 fill!(EV, 0.0)
    #                 fillflows!(ubV, flow, p, θ2, σ, i, itype...)
    #                 ubV .+= β .* EV1
    #                 ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, tmpv_vw, p; vftol=1e-10, maxit0=20, maxit1=40)
    #                 fdEV0[:,:,k] .+= EV0
    #                 fdEV0[:,:,k] ./= hh
    #             end
    #
    #             fill!(EV, 0.0)
    #             fill!(dEV, 0.0)
    #             fillflows!(ubV, flow, p, θt, σ, i, itype...)
    #             fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
    #             ubV .+= β .* EV1
    #             dubV .+= β .* dEV1
    #             ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, tmpv_vw, p; vftol=1e-10, maxit0=20, maxit1=40)
    #             ubV[:,:,1] .= β .* EV0
    #             ShaleDrillingModel.gradinf!(dEV0, tmpv_vw, p)   # note: destroys ubV & dubV
    #
    #             println("extrema(dEV0) = $(extrema(dEV0))")
    #             @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
    #             @views sub = CartesianIndices(fdEV0)[idx]
    #             println("worst value is $maxv at $sub for dθ")
    #
    #             @test 0.0 < maxv < 1.0
    #             @test all(isfinite.(dEV0))
    #             @test all(isfinite.(fdEV0))
    #             @test fdEV0 ≈ dEV0
    #
    #         end
    #     end
    # end









































end # testset
end # module
