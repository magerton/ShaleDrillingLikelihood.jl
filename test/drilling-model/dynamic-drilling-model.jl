module ShaleDrillingLikelihood_DynamicDrillModelTest

# using Revise

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
    ubVfull    ,
    dubVfull   ,
    dubVfullperm,
    q          ,
    lse        ,
    tmp        ,
    tmp_cart   ,
    Πψtmp      ,
    IminusTEVp,
    ΠsumdubVj,
    dev0tmpj,
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
    tmpEVj,
    dEV,
    EV,
    anticipate_t1ev,
    _dmax,
    solve_inf_vfit!,
    solve_inf_pfit!,
    solve_inf_vfit_pfit!,
    gradinf!,
    solve_vf_terminal!,
    solve_vf_infill!,
    learningUpdate!,
    solve_vf_explore!,
    solve_vf_all!,
    solve_vf_all_timing!,
    solve_vf_infill_timing!,
    _nSexp,
    update_static_payoffs!,
    NoValueFunction,
    ValueFunction,
    ValueFunctionArrayOnly,
    value_function

println("print to keep from blowing up")

@testset "Dynamic Drilling Model" begin

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant(), ScrapValue_Constant())
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
    ddm_no_t1ev   = DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, false)
    ddm_with_t1ev = DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, true)

    if false
        @code_warntype ValueFunction(f, 0.9, wp, zs, ztrans, ψs)
        @code_warntype DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, true, ValueFunction)
        @code_warntype DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, true, ValueFunctionArrayOnly)
        @code_warntype DynamicDrillModel(f, 0.9, wp, zs, ztrans, ψs, true, NoValueFunction)
    end

    @testset "VF Structs" begin
        vfao = ValueFunctionArrayOnly(ddm_no_t1ev)
        vf = ValueFunction(ddm_no_t1ev)
        @test EV(vfao) == EV(vf)
        @test dEV(vfao) == dEV(vf)
    end

    tmpv = DCDPTmpVars(ddm_no_t1ev)
    @test tmpv isa ShaleDrillingLikelihood.DCDPTmpVarsArray
    @test view(tmpv, 1:2) isa ShaleDrillingLikelihood.DCDPTmpVarsView

    fd = similar(dubVfull(tmpv))
    ubvminus = zero(ubVfull(tmpv))
    ubvplus  = zero(ubVfull(tmpv))
    thetaminus = similar(theta)
    thetaplus = similar(theta)

    @testset "Fill per-period flow payoffs" begin

        ddm = ddm_with_t1ev
        tmpvminus = DCDPTmpVars(ubvminus, dubVfull(tmpv), dubVfullperm(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv), tmpEVj(tmpv), ΠsumdubVj(tmpv), dev0tmpj(tmpv) )
        tmpvplus  = DCDPTmpVars(ubvplus,  dubVfull(tmpv), dubVfullperm(tmpv), q(tmpv), lse(tmpv), tmp(tmpv), tmp_cart(tmpv), Πψtmp(tmpv), IminusTEVp(tmpv), tmpEVj(tmpv), ΠsumdubVj(tmpv), dev0tmpj(tmpv) )

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

                update_static_payoffs!(view(tmpvminus, idxd), ddm, thetaminus, sidx, ichar, false)
                update_static_payoffs!(view(tmpvplus , idxd), ddm, thetaplus, sidx, ichar, false)
                @test all(dubVfull(tmpv) .== 0)

                fd[i,:,:,:] .= (ubVfull(tmpvplus) .- ubVfull(tmpvminus))./ twoh
            end

            update_static_payoffs!(view(tmpv, idxd), ddm, theta, sidx, ichar, true)

            fdnosig = fd[1:end-1,:,:,:] .- dubVfull(tmpv)[1:end-1,:,:,:]
            fdsig = fd[end,:,:,:] .- dubVfull(tmpv)[end,:,:,:]

            if PRINTSTUFF
                @views maxv, idx = findmax(abs.(fdnosig))
                println("worst value is $maxv at $(CartesianIndices(fdnosig)[idx])")

                @views maxv, idx = findmax(abs.(fdsig))
                println("worst value is $maxv at $(CartesianIndices(fdsig)[idx])")
            end

            @test fd ≈ dubVfull(tmpv)
        end

        if DOBTIME
            @btime update_static_payoffs!($tmpv, $ddm, $theta, 2, (2.0, 0.25,), true)
        end
    end

    @testset "Learning transition" begin
        ddm = ddm_with_t1ev
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

            ddm = ddm_with_t1ev
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
            PRINTSTUFF && println("worst value is $maxv at $sub for β dΠ/dσ")

            @test 0.0 < maxv < 1e-5
            @test fdpi ≈ pipsi

        end
    end

    @testset "VF interpolation" begin



    end

    @testset "VF Iteration" begin

        thetaminus = similar(theta)
        thetaplus = similar(theta)

        for ddm in (ddm_with_t1ev, ddm_no_t1ev)

            evs = ValueFunction(ddm)
            fdEV = zero(dEV(evs))

            # println("\n-----------------------\nanticipate_e = $(anticipate_t1ev(ddm))\n---------------------------------------")

            @testset "Finite horizon VF gradient with anticipate_e = $(anticipate_t1ev(ddm))" begin
                for i in 1:length(statespace(ddm))
                    PRINTSTUFF && println("state $i of $(length(statespace(ddm)))")
                    idxd  = dp1space(wp,i)
                    idxs  = collect(sprimes(statespace(ddm),i))
                    horzn = _horizon(wp,i)

                    tmp_vw = view(tmpv, idxd)

                    EV0   = view(EV(evs) ,    :, :, i)
                    EV1   = view(EV(evs) ,    :, :, idxs)
                    dEV0  = view(dEV(evs), :, :, :, i)
                    dEV1  = view(dEV(evs), :, :, :, idxs)
                    fdEV0 = view(fdEV    , :, :, :, i)

                    fill!(fdEV0, 0)

                    for k in OneTo(length(theta))
                        thetaminus .= theta
                        thetaplus .= theta
                        h = peturb(theta[k])
                        thetaminus[k] -= h
                        thetaplus[k] += h
                        hh = thetaplus[k] - thetaminus[k]

                        fill!(evs, 0)
                        update_static_payoffs!(tmp_vw, ddm, thetaminus, i, ichar, false)
                        vfit!(EV0, tmp_vw, ddm)
                        ubV(tmp_vw) .+= discount(ddm) .* EV1
                        fdEV0[:,:,k] .-= EV0

                        fill!(evs, 0)
                        update_static_payoffs!(tmp_vw, ddm, thetaplus, i, ichar, false)
                        vfit!(EV0, tmp_vw, ddm)
                        ubV(tmp_vw) .+= discount(ddm) .* EV1
                        fdEV0[:,:,k] .+= EV0

                        fdEV0[:,:,k] ./= hh
                    end

                    fill!(evs, 0)
                    update_static_payoffs!(tmp_vw, ddm, theta, i, ichar, true)
                    ubV(tmp_vw)  .+= discount(ddm) .* EV1
                    dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
                    vfit!(EV0, dEV0, tmp_vw, ddm)

                    PRINTSTUFF && println("\textrema(dEV0) = $(extrema(dEV0))")
                    PRINTSTUFF && println("\textrema(fdEV0) = $(extrema(fdEV0))")

                    @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
                    @views sub = CartesianIndices(fdEV0)[idx]
                    PRINTSTUFF && println("\tworst value is $maxv at $sub for dθ")

                    @test 0.0 <= maxv < 1.0
                    @test all(isfinite.(dEV0))
                    @test all(isfinite.(fdEV0))
                    @test fdEV0 ≈ dEV0
                end
            end # test finite horizon

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
                update_static_payoffs!(tmp_vw, ddm, theta, i, ichar, true)
                ubV(tmp_vw)  .+= discount(ddm) .* EV1
                dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
                converged, iterv, bnds = solve_inf_vfit!(EV0, tmp_vw, ddm; maxit=1000, vftol=1e-10)
                PRINTSTUFF && println("vfit done. converged = $converged after $iterv iterations. error bounds are $bnds")
                vfEV0 .= EV0

                fill!(evs, 0)
                update_static_payoffs!(tmp_vw, ddm, theta, i, ichar, true)
                ubV(tmp_vw)  .+= discount(ddm) .* EV1
                dubVperm(tmp_vw) .+= discount(ddm) .* dEV1
                converged, iterp, bnds = solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-10, maxit0=20, maxit1=40)
                PRINTSTUFF && println("pfit done. converged = $converged after $iterp iterations. error bounds are $bnds")

                @test iterv > iterp

                @views maxv, idx = findmax(abs.(vfEV0 .- EV0))
                @views sub = CartesianIndices(EV0)[idx]
                PRINTSTUFF &&  println("worst value is $maxv at $sub for vfit vs pfit")
                @test EV0 ≈ vfEV0

            end # test infinite horizon


            @testset "Check gradient for infinite horizon, anticipate_e = $(anticipate_t1ev(ddm))" begin

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

                nk = length(theta)
                for k in OneTo(nk)
                    thetaminus .= theta
                    thetaplus .= theta
                    h = peturb(theta[k])
                    thetaminus[k] -= h
                    thetaplus[k] += h
                    hh = thetaplus[k] - thetaminus[k]

                    fill!(evs, 0)
                    update_static_payoffs!(tmp_vw, ddm, thetaminus, i, ichar, false)
                    ubV(tmp_vw) .+= discount(ddm) .* EV1
                    solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-11, maxit0=10, maxit1=40)
                    fdEV0[:,:,k] .-= EV0

                    fill!(evs, 0)
                    update_static_payoffs!(tmp_vw, ddm, thetaplus, i, ichar, false)
                    ubV(tmp_vw) .+= discount(ddm) .* EV1
                    solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-11, maxit0=10, maxit1=40)
                    fdEV0[:,:,k] .+= EV0

                    fdEV0[:,:,k] ./= hh
                end
                @test any(fdEV0 .!= 0)


                fill!(evs, 0.0)
                update_static_payoffs!(tmp_vw, ddm, theta, i, ichar, true)
                ubV(tmp_vw)      .+= discount(ddm) .* EV1
                dubVperm(tmp_vw) .+= discount(ddm) .* dEV1

                solve_inf_vfit_pfit!(EV0, tmp_vw, ddm; vftol=1e-11, maxit0=10, maxit1=40)
                ubV(tmp_vw)[:,:,1] .= discount(ddm) .* EV0
                gradinf!(dEV0, tmp_vw, ddm)   # note: destroys ubV & dubV
                @test any(dEV0 .!= 0)

                PRINTSTUFF && println("extrema(dEV0) = $(extrema(dEV0))")
                PRINTSTUFF && println("extrema(fdEV0) = $(extrema(fdEV0))")
                @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
                @views sub = CartesianIndices(fdEV0)[idx]
                PRINTSTUFF && println("worst value is $maxv at $sub for dθ")

                @test 0.0 < maxv < 1.0
                @test all(isfinite.(dEV0))
                @test all(isfinite.(fdEV0))
                @test fdEV0 ≈ dEV0

            end


            # ---------------- Regime 1 VFI and PFI - test gradient ------------------

            @testset "Gradient of EV for infill, anticipate_e = $(anticipate_t1ev(ddm))" begin
                θ1 = similar(theta)
                θ2 = similar(theta)
                fdEV = similar(dEV(evs))
                itype = (4.7, 0.25,)

                nk = length(theta)

                for k = 1:nk
                    h = peturb(theta[k])
                    θ1 .= theta
                    θ2 .= theta
                    θ1[k] -= h
                    θ2[k] += h
                    hh = θ2[k] - θ1[k]

                    fill!(evs, 0)
                    solve_vf_terminal!(ddm, false)
                    @test all(EV(evs) .== 0)
                    @test all(dEV(evs) .== 0)
                    solve_vf_infill!(tmpv, ddm, θ1, itype, false)
                    fdEV[:,:,k,:] .= -EV(evs)

                    @test any( EV(evs) .!= 0)
                    @test all(dEV(evs) .== 0)

                    solve_vf_terminal!(ddm, false)
                    solve_vf_infill!(tmpv, ddm, θ2, itype, false)
                    fdEV[:,:,k,:] .+= EV(evs)
                    fdEV[:,:,k,:] ./= hh
                end

                solve_vf_terminal!(ddm, true)
                solve_vf_infill!(tmpv, ddm, theta, itype, true)
                @test any(dEV(evs) .!= 0)


                @views maxv, idx = findmax(abs.(fdEV[:,:,2:end,:].-dEV(evs)[:,:,2:end,:]))
                @views sub = CartesianIndices(fdEV[:,:,2:end,:])[idx]
                PRINTSTUFF && println("worst value is $maxv at $sub for dθ[2:end]")

                maxv, idx = findmax(abs.(fdEV.-dEV(evs)))
                sub = CartesianIndices(fdEV)[idx]
                PRINTSTUFF && println("worst value is $maxv at $sub")
                @test 0.0 < maxv < 1.0
                @test all(isfinite.(dEV(evs)))
                @test all(isfinite.(fdEV))
                @test fdEV ≈ dEV(evs)
            end




            @testset "Gradient of EV for Exploratory + learning, anticipate_e = $(anticipate_t1ev(ddm))" begin

                θ1 = similar(theta)
                θ2 = similar(theta)
                fdEV = similar(dEV(evs))
                itype = (4.7, 0.25,)

                nsexp = _nSexp(statespace(ddm))
                nk = length(theta)

                for k = 1:nk
                    h = peturb(theta[k])
                    θ1 .= theta
                    θ2 .= theta
                    θ1[k] -= h
                    θ2[k] += h
                    hh = θ2[k] - θ1[k]

                    fill!(evs, 0)

                    solve_vf_all!(tmpv, ddm, θ1, itype, false)
                    fdEV[:,:,k,:] .= -EV(evs)

                    solve_vf_all!(tmpv, ddm, θ2, itype, false)
                    fdEV[:,:,k,:] .+= EV(evs)

                    fdEV[:,:,k,:] ./= hh
                end

                # ----------------- analytic -----------------

                solve_vf_all!(tmpv, ddm, theta, itype, true)

                # check infill + learning portion gradient
                @views a =     fdEV[:,:, :, nsexp+1:end]
                @views b = dEV(evs)[:,:, :, nsexp+1:end]
                maxv, idx = findmax(abs.(a .- b))
                PRINTSTUFF && println("worst value is $maxv at $(CartesianIndices(a)[idx]) for infill")

                # check exploration portion of gradient
                @views a =     fdEV[:,:, 1:end-1, 1:nsexp]
                @views b = dEV(evs)[:,:, 1:end-1, 1:nsexp]
                maxv, idx = findmax(abs.(a .- b))
                PRINTSTUFF && println("worst value is $maxv at $(CartesianIndices(a)[idx]) for infill")

                # check exploration portion of gradient
                @views a =     fdEV[:,:, end, 1:nsexp]
                @views b = dEV(evs)[:,:, end, 1:nsexp]
                maxv, idx = findmax(abs.(a .- b))
                PRINTSTUFF && println("worst value is $maxv at $(CartesianIndices(a)[idx]) for infill")

                @test 0.0 < maxv < 0.1
                @test all(isfinite.(dEV(evs)))
                @test all(isfinite.(fdEV))

                @test fdEV ≈ dEV(evs)
                # println("dEV/dθ looks ok! :)")
            end

            if DOBTIME
                itype = (4.7, 0.25,)
                println("Timing solve_vf_all! with anticipate_e = $(anticipate_t1ev(ddm))")
                fill!(value_function(ddm), 0)
                @btime solve_vf_terminal!(            $ddm,                 true)
                @btime solve_vf_infill_timing!($tmpv, $ddm, $theta, $itype, true)
                @btime learningUpdate!(        $tmpv, $ddm, $theta, $itype, true)
                @btime solve_vf_explore!(      $tmpv, $ddm, $theta, $itype, true)

                @btime solve_vf_all_timing!($tmpv, $ddm, $theta, $itype, true)
            end

            if DOPROFILE
                println("Profiling solve_vf_all_timing! with anticipate_e = $(anticipate_t1ev(ddm))")
                itype = (4.7, 0.25,)

                @code_warntype solve_vf_terminal!(      ddm,               true)
                @code_warntype solve_vf_infill!(    tmpv, ddm, theta, itype, true)
                @code_warntype learningUpdate!(     tmpv, ddm, theta, itype, true)
                @code_warntype solve_vf_explore!(   tmpv, ddm, theta, itype, true)
                fill!(evs, 0)
                @code_warntype solve_vf_all!(       tmpv, ddm, theta, itype, true)
                @code_warntype solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                fill!(evs, 0)
                # solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # Profile.clear()
                # @profile solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # @profile solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # @profile solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # @profile solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # @profile solve_vf_all_timing!(tmpv, ddm, theta, itype, true)
                # Juno.profiletree()
                # Juno.profiler()
                # Profile.print(format=:flat)
                # ProfileView.view()
                # pprof()
            end

        end # ddm
    end # VF Iteration



end # testset
end # module
