module ShaleDrillingLikelihood_OverallLikelihood_Test

# using Revise
# using Juno
# using Profile
# using ProfileView
# using PProf

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using InteractiveUtils
using LinearAlgebra
using BenchmarkTools

using Base: OneTo

using ShaleDrillingLikelihood: AbstractDataSet,
    DataRoyalty, DataProduce, DataDrill,
    TestDrillModel,
    simloglik!,
    SimulationDraws,
    idx_drill_ρ, idx_royalty_ρ,
    idx_drill_ψ, idx_produce_ψ,
    _nparm,
    thetas

println("testing overall likelihood")

@testset "Testing Joint Likelihoods" begin
    num_i = 1_200
    L = 3
    M = 1_000
    u = randn(num_i)
    v = randn(num_i)
    psi2 = u

    # set up coefs
    θρ = 0.5
    αψ = 0.6

    # set up drilling coefs
    model_drill = TestDrillModel()
    θ_drill   = [αψ, 2.0, -2.0, -0.75, θρ]
    θ_royalty = [θρ, 1.0,    -1.0, 1.0, 1.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ
    θ_produce = vcat(αψ, rand(3), 0.3, 0.4)

    θ = vcat(θ_drill, θ_royalty[2:end], θ_produce[2:end])
    coef_links = [(idx_produce_ψ, idx_drill_ψ,),]

    # make data
    data_drill_opt = (num_zt=200, minmaxleases=1:2, nper_initial=10:20, nper_development=0:10, tstart=1:50,)
    data_drill   = DataDrill(u, v, TestDrillModel(), θ_drill; data_drill_opt...)
    data_produce = DataProduce(u, 10, 10:20, θ_produce)
    data_royalty = DataRoyalty(u,v,θ_royalty,L)
    data_full = DataSetofSets(data_drill, data_royalty, data_produce, coef_links)
    sim = SimulationDraws(M, data_drill)

    @testset "Joint likelihood of Royalty + Pdxn" begin

        println("testing joint likelihood of royalty + pdxn")
        data = DataSetofSets(EmptyDataSet(), data_royalty, data_produce)
        theta = vcat(θ_royalty, θ_produce)
        grad = similar(theta)
        nparm = _nparm(data)
        hess = Matrix{eltype(theta)}(undef, nparm, nparm)
        tmpgrads = Matrix{eltype(theta)}(undef, nparm, num_i)

        simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        simloglik!(grad, hess, tmpgrads, data, theta, sim, true)

        fd = Calculus.gradient(xx -> simloglik!(grad, hess, tmpgrads, data, xx, sim, false), theta, :central)

        fill!(grad,0)
        fill!(hess,0)
        simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
        @test !all(grad.==0)
        @test isapprox(fd, grad; rtol=2e-5)

        print("")
        @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, false)
        @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, true)
        print("")

        # @code_warntype simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        # println("\n\n\n----------------------------\n\n\n")
        # @code_warntype simloglik!(grad, hess, tmpgrads, data, theta, sim, true)

        # Profile.clear()
        # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        # # Juno.profiletree()
        # # Juno.profiler()
        # Profile.print(format=:flat)
        # # ProfileView.view()
        # pprof()

        # Profile.clear()
        # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
        # ProfileView.view()
    end

    @testset "Joint likelihood of Drill + Roy + Pdxn" begin

        println("testing joint likelihood of FULL data!")
        data = data_full
        theta = θ
        grad = similar(theta)
        nparm = _nparm(data)
        hess = Matrix{eltype(theta)}(undef, nparm, nparm)
        tmpgrads = Matrix{eltype(theta)}(undef, nparm, num_i)

        simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        simloglik!(grad, hess, tmpgrads, data, theta, sim, true)

        fd = Calculus.gradient(xx -> simloglik!(grad, hess, tmpgrads, data, xx, sim, false), theta, :central)

        fill!(grad,0)
        fill!(hess,0)
        simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
        @test !all(grad.==0)
        @test isapprox(fd, grad; rtol=2e-5)

        print("")
        @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, false)
        @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, true)
        print("")

        # @code_warntype simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        # println("\n\n\n----------------------------\n\n\n")
        # @code_warntype simloglik!(grad, hess, tmpgrads, data, theta, sim, true)

        # Profile.clear()
        # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
        # # Juno.profiletree()
        # # Juno.profiler()
        # Profile.print(format=:flat)
        # # ProfileView.view()
        # pprof()

        # Profile.clear()
        # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
        # ProfileView.view()
    end

end

end # module
