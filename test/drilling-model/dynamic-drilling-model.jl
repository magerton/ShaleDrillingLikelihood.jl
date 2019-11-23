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
    dubVfullperm,
    q          ,
    lse        ,
    tmp        ,
    tmp_cart   ,
    Πψtmp      ,
    IminusTEVp,
    ubV,
    dubV,
    dubVperm,
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
    _ρ,
    dp1space,
    sprimes,
    _horizon,
    vfit!,
    dEV,
    EV,
    anticipate_t1ev,
    _dmax,
    solve_inf_vfit!,
    solve_inf_pfit!,
    solve_inf_vfit_pfit!

const DOBTIME = false

println("print")

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
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (2.0, 0.25,)

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

        tmpvminus = DCDPTmpVars(ubvminus, dubVfull(tmpv), dubVfullperm(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv))
        tmpvplus  = DCDPTmpVars(ubvplus,  dubVfull(tmpv), dubVfullperm(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv))

        for sidx in 1:length(wp)
            T = eltype(theta)
            idxd = dp1space(wp,sidx)

            for i in OneTo(length(theta))
                thetaminus .= theta
                thetaplus .= theta
                h = peturb(theta[i])
                thetaplus[i] += h
                thetaminus[i] -= h
                twoh = thetaplus[i] - thetaminus[i]
                @test maximum(abs.(thetaplus .- thetaminus)) ≈ twoh

                fill!(dubVfull(tmpv), 0)
                fill!(ubVfull(tmpvminus), 0)
                fill!(ubVfull(tmpvplus), 0)

                flow!(view(tmpvminus, idxd), ddm, thetaminus, sidx, ichar, false)
                flow!(view(tmpvplus , idxd), ddm, thetaplus, sidx, ichar, false)
                @test all(dubVfull(tmpv) .== 0)

                fd[i,:,:,:] .= (ubVfull(tmpvplus) .- ubVfull(tmpvminus))./ twoh
            end

            flow!(view(tmpv, idxd), ddm, theta, sidx, ichar, true)

            fdnosig = fd[1:end-1,:,:,:] .- dubVfull(tmpv)[1:end-1,:,:,:]
            fdsig = fd[end,:,:,:] .- dubVfull(tmpv)[end,:,:,:]

            # @views maxv, idx = findmax(abs.(fdnosig))
            # println("worst value is $maxv at $(CartesianIndices(fdnosig)[idx])")
            #
            # @views maxv, idx = findmax(abs.(fdsig))
            # println("worst value is $maxv at $(CartesianIndices(fdsig)[idx])")

            @test fd ≈ dubVfull(tmpv)
        end

        if DOBTIME
            @btime flow!($tmpv, $ddm, $theta, 2, (2.0, 0.25,), true)
        end
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

            pipsi = Πψtmp(tmpv)
            fdpi = similar(pipsi)
            @test fdpi isa Matrix
            @test !(fdpi === pipsi)
            pi1 = similar(fdpi)
            pi2 = similar(fdpi)

            _βΠψ!(tmpv, ddm, theta1)
            pi1 .= pipsi
            _βΠψ!(tmpv, ddm, theta2)
            pi2 .= pipsi

            fdpi .= (pi2 .- pi1) ./ hh
            @test any(fdpi .!= 0)
            _βΠψdθρ!(tmpv, ddm, theta)

            @views maxv, idx = findmax(abs.(pipsi .- fdpi))
            sub = CartesianIndices(fdpi)[idx]
            # @show "worst value is $maxv at $sub for β dΠ/dσ"

            @test 0.0 < maxv < 1e-5
            @test fdpi ≈ pipsi

        end
    end

    @testset "VF Iteration" begin

        @testset "Finite horizon VF gradient" begin

            fdEV = zero(dEV(evs))
            thetaminus = similar(theta)
            thetaplus = similar(theta)

            for i in 1:length(statespace(ddm))

                idxd  = dp1space(wp,i)
                idxs  = collect(sprimes(statespace(ddm),i))
                horzn = _horizon(wp,i)

                tmp_vw = view(tmpv, idxd)

                EV0   = view(EV(evs) ,    :, :, i)
                EV1   = view(EV(evs) ,    :, :, idxs)
                dEV0  = view(dEV(evs), :, :, :, i)
                dEV1  = view(dEV(evs), :, :, :, idxs)
                fdEV0 = view(fdEV    , :, :, :, i)

                for k in OneTo(length(theta))
                    thetaminus .= theta
                    thetaplus .= theta
                    h = peturb(theta[k])
                    thetaminus[k] -= h
                    thetaplus[k] += h
                    hh = thetaplus[k] - thetaminus[k]

                    fill!(evs, 0)
                    flow!(tmp_vw, ddm, thetaminus, i, ichar, false)
                    vfit!(EV0, tmp_vw, ddm)
                    ubV(tmp_vw) .+= discount(ddm) .* EV0
                    fdEV0[:,:,k] .-= EV0

                    fill!(evs, 0)
                    flow!(tmp_vw, ddm, thetaplus, i, ichar, false)
                    vfit!(EV0, tmp_vw, ddm)
                    ubV(tmp_vw) .+= discount(ddm) .* EV0
                    fdEV0[:,:,k] .+= EV0

                    fdEV0[:,:,k] ./= hh
                end

                fill!(evs, 0)
                flow!(tmp_vw, ddm, theta, i, ichar, true)
                ubV(tmp_vw)  .+= discount(ddm) .* EV0
                dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
                vfit!(EV0, dEV0, tmp_vw, ddm)

                # println("extrema(dEV0) = $(extrema(dEV0))")
                # println("extrema(fdEV0) = $(extrema(fdEV0))")

                @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
                @views sub = CartesianIndices(fdEV0)[idx]
                # println("worst value is $maxv at $sub for dθ")

                @test 0.0 <= maxv < 1.0
                @test all(isfinite.(dEV0))
                @test all(isfinite.(fdEV0))
                @test fdEV0 ≈ dEV0
            end
        end # finite horizon



        @testset "Check vfit/pfit for infinite horizon, anticipate_e = $(anticipate_t1ev(ddm))" begin
            i = length(statespace(ddm))-1

            fdEV = zero(dEV(evs))

            idxd  = dp1space(wp,i)
            idxs  = collect(sprimes(statespace(ddm),i))
            horzn = _horizon(wp,i)
            @test horzn == :Infinite

            EV0   = view(EV(evs) ,    :, :, i)
            EV1   = view(EV(evs) ,    :, :, idxs)
            dEV0  = view(dEV(evs), :, :, :, i)
            dEV1  = view(dEV(evs), :, :, :, idxs)
            fdEV0 = view(fdEV    , :, :, :, i)

            tmp_vw = view(tmpv, idxd)

            vfEV0 = zeros(size(EV0))

            fill!(evs, 0)
            flow!(tmp_vw, ddm, theta, i, ichar, true)
            ubV(tmp_vw)  .+= discount(ddm) .* EV0
            dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
            converged, iter, bnds = solve_inf_vfit!(EV0, tmp_vw, ddm; maxit=1000, vftol=1e-10)
            println("vfit done. converged = $converged after $iter iterations. error bounds are $bnds")
            vfEV0 .= EV0

            fill!(evs, 0)
            flow!(tmp_vw, ddm, theta, i, ichar, true)
            ubV(tmp_vw)  .+= discount(ddm) .* EV0
            dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
            converged, iter, bnds = solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-10, maxit0=20, maxit1=40)
            println("pfit done. converged = $converged after $iter iterations. error bounds are $bnds")

            @views maxv, idx = findmax(abs.(vfEV0 .- EV0))
            @views sub = CartesianIndices(EV0)[idx]
            println("worst value is $maxv at $sub for vfit vs pfit")
            @test EV0 ≈ vfEV0
        end
    end # vfit



end # testset
end # module
