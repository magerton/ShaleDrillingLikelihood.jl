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

using ShaleDrillingLikelihood: DataRoyalty, DataProduce,
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
    idx_theta


println("testing overall royalty")

@testset "Joint likelihood of Royalty + Pdxn" begin

    num_i = 500
    L = 3
    M = 500
    u = randn(num_i)
    v = randn(num_i)
    psi2 = u

    sim = SimulationDraws(M, num_i)

    theta_produce = vcat(0.5, rand(3), 0.3, 0.4)
    theta_royalty = [0.1, 1.0,    -2.0, 2.0, 2.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ

    data_produce = DataProduce(psi2, 10, 10:20, theta_produce)
    data_royalty = DataRoyalty(u,v,theta_royalty,L)
    @test length(data_royalty) == num_i

    data = (data_royalty, data_produce)
    @test idx_theta(data) == (OneTo(length(theta_royalty)), length(theta_royalty) .+ OneTo(length(theta_produce)))

    nparm = sum(_nparm.(data))
    theta = vcat(theta_royalty, theta_produce)
    grad = similar(theta)
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


    # @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, false)
    # @show @benchmark simloglik!($grad, $hess, $tmpgrads, $data, $theta, $sim, true)

    # Profile.clear()
    # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, false)
    # ProfileView.view()

    # Profile.clear()
    # @profile simloglik!(grad, hess, tmpgrads, data, theta, sim, true)
    # ProfileView.view()

end

end # module
