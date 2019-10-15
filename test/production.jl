module ShaleDrillingLikelihood_Production_Test

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using LinearAlgebra
using BenchmarkTools

using ShaleDrillingLikelihood: _num_x,
    idx_produce_ψ, idx_produce_β, idx_produce_σ2η, idx_produce_σ2u,
    theta_produce, theta_produce_ψ, theta_produce_β, theta_produce_σ2η, theta_produce_σ2u,
    simloglik_produce!, grad_simloglik_produce!, loglik_produce_scalars,
    DataProduce, ObservationGroupProduce, ObservationProduce,
    _x, _y, _xsum, _nu, _i, update_nu!, update_xpnu!, _qm, _psi2,
    SimulationDraws, psi2_wtd_sum_and_sumsq, Observation,
    _nparm

println("Starting production likelihood tests")

@testset "Production simulated likelihood" begin

    Random.seed!(1234)

    k = 3
    beta = rand(k)
    sigmas = (0.5, 0.3, 0.4,)
    theta = vcat(sigmas[1], beta, sigmas[2:3]...)
    num_i = 100
    M = 500

    data = DataProduce(num_i, 10, 10:20, theta)
    allsim = SimulationDraws(M,num_i)

    @test length(theta) == _nparm(data)
    @test _num_x(data) == k

    @test     theta_produce_ψ(  data,theta)  == theta[1]
    @test all(theta_produce_β(  data,theta) .== theta[1 .+ (1:k)])
    @test     theta_produce_σ2η(data,theta)  == theta[end-1]
    @test     theta_produce_σ2u(data,theta)  == theta[end]

    ff(x)          = grad_simloglik_produce!(zeros(length(x)), data, x, allsim, false)
    ffgg!(grad, x) = grad_simloglik_produce!(grad,             data, x, allsim, true)

    tmpgrad = similar(theta)
    ff(theta)

    fd = Calculus.gradient(ff, theta)
    fill!(tmpgrad, 0)
    ffgg!(tmpgrad, theta)
    # @test fd ≈ tmpgrad
    @test norm(fd .- tmpgrad, Inf) < 2e-5

    od = OnceDifferentiable(ff, ffgg!, ffgg!, theta)
    res = optimize(od, theta, BFGS(), Optim.Options(time_limit = 40.0))
    @test norm(res.minimizer .- theta, Inf) < 0.02
end


println("Beginning speed tests")

M = 500
num_i = 2000
sigmas = (0.5, 0.3, 0.4,)
beta = randn(3)
theta = vcat(sigmas[1], beta, sigmas[2:3]...)
data = DataProduce(num_i, 10, 10:20, theta)
sim = SimulationDraws(M, num_i)
simi = view(sim, 1)
_qm(sim) .= softmax(randn(M))

# @show @benchmark psi2_wtd_sum_and_sumsq(simi)

obs = Observation(data,1)
# @show @benchmark simloglik_produce!(obs, theta, simi)


end # module
