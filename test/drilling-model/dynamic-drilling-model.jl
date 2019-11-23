module ShaleDrillingLikelihood_DynamicDrillingModelTest

using ShaleDrillingLikelihood
using Test
using SparseArrays
using BenchmarkTools

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
    end_ex0

@testset "Dynamic Drilling Model" begin

    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    theta = randn(_nparm(f))


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

    for sidx in 1:length(wp)
        T = eltype(theta)
        for i in OneTo(length(theta))
            thetaminus .= theta
            thetaplus .= theta
            h = max( abs(theta[i]), one(T) ) * cbrt(eps(T))
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

end # testset
end # module
