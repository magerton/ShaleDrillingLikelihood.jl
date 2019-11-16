module ShaleDrillingLikelihood_OverallLikelihood_Test

# using Revise
# using Juno
# using Profile
# using ProfileView

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
    simloglik_royalty!,
    grad_simloglik_royalty!,
    simloglik_produce!,
    grad_simloglik_produce!,
    simloglik!,
    update_ψ1!,
    update_dψ1dρ!,
    update_xpnu!,
    update_nu!,
    SimulationDraws,
    _nparm,
    ObservationGroup,
    ObservationGroupProduce,
    ObservationRoyalty,
    TestDrillModel,
    theta_drill_ρ,
    theta_drill_ψ,
    theta_royalty_ρ,
    theta_produce_ψ,
    idx_drill, theta_drill,
    idx_royalty, theta_royalty,
    idx_produce, theta_produce,
    idx_drill_ρ, idx_royalty_ρ,
    idx_drill_ψ, idx_produce_ψ,
    thetas

println("testing overall royalty")

@testset "Joint likelihood of Royalty + Pdxn" begin

    num_i = 500
    L = 3
    M = 500
    u = randn(num_i)
    v = randn(num_i)
    psi2 = u

    sim = SimulationDraws(M, num_i)

    # set up coefs
    θρ = 0.5
    αψ = 0.6

    # set up drilling coefs
    model_drill = TestDrillModel()
    θ_drill   = [αψ, 2.0, -2.0, -0.75, θρ]
    θ_royalty = [θρ, 1.0,    -1.0, 1.0, 1.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ
    θ_produce = vcat(αψ, rand(3), 0.3, 0.4)

    θ = vcat(θ_drill, θ_royalty[2:end], θ_produce[2:end])

    # make data
    data_drill_opt = (num_zt=200, minmaxleases=1:2, nper_initial=10:20, nper_development=0:10, tstart=1:50,)
    data_drill   = DataDrill(u, v, TestDrillModel(), θ_drill; data_drill_opt...)
    data_produce = DataProduce(u, 10, 10:20, θ_produce)
    data_royalty = DataRoyalty(u,v,θ_royalty,L)

    data = DataSetofSets(data_drill, data_royalty, data_produce)

    # theta = vcat(θ_royalty, θ_produce)
    # grad = similar(theta)
    # hess = Matrix{eltype(theta)}(undef, nparm, nparm)
    # tmpgrads = Matrix{eltype(theta)}(undef, nparm, num_i)
    #
    # simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
    # simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
    #
    # fd = Calculus.gradient(xx -> simloglik!(grad, hess, tmpgrads, data, xx, sim, false), theta, :central)
    #
    # fill!(grad,0)
    # fill!(hess,0)
    # simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
    # @test !all(grad.==0)
    # @test isapprox(fd, grad; rtol=2e-5)
    #
    #
    # @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, false)
    # @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, true)
    #
    # Profile.clear()
    # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
    # ProfileView.view()
    #
    # Profile.clear()
    # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
    # ProfileView.view()

end

end # module
