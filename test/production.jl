module ShaleDrillingLikelihood_Production_Test

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using LinearAlgebra


using ShaleDrillingLikelihood: _num_x,
    idx_produce_ψ, idx_produce_β, idx_produce_σ2η, idx_produce_σ2u,
    theta_produce, theta_produce_ψ, theta_produce_β, theta_produce_σ2η, theta_produce_σ2u,
    ProductionLikelihoodComputations, ProductionGradientComputations,
    grad_simloglik_produce!, loglik_produce, loglik_produce_scalars,
    DataProduce, ObservationGroupProduce, ObservationProduce,
    _x, _y, _xsum, _nu, _i, update_nu!


@testset "Production basics" begin

    Random.seed!(1234)

    # Generate dataset where ψ=1
    k = 3
    num_i = 500
    num_t = 11
    nobs = num_i * num_t
    pm = ProductionModel()
    M = 1

    theta = [1.0, 2.0, 3.0, 4.0, sqrt(0.10), sqrt(0.10)]
    X = randn(k,nobs)
    y = theta[1] .+ X'*theta[2:end-2]

    grpptr = collect(1:num_i+1)
    obsptr = collect(1:num_t:(num_i*num_t+1))

    data = DataProduce(y,X,obsptr, grpptr)

    @test length(theta) == length(pm,data)
    @test _num_x(pm,data) == k

    @test     theta_produce_ψ(  pm,data,theta)  == theta[1]
    @test all(theta_produce_β(  pm,data,theta) .== theta[1 .+ (1:k)])
    @test     theta_produce_σ2η(pm,data,theta)  == theta[end-1]
    @test     theta_produce_σ2u(pm,data,theta)  == theta[end]

    # need to add in shocks
    for g in data
        for o in g
            u = theta_produce_σ2u(pm,data,theta)*randn()
            _y(o) .+= u .+ theta_produce_σ2η(pm,data,theta).*randn(length(o))
        end
    end

    function fg!(grad, θ, yy, xx, dograd::Bool=true)
        llm = zeros(Float64, M)
        qm = llm
        vi = zeros(Float64, num_t)

        update_nu!(data, pm, θ)

        LL = 0.0

        for grp in data
            for obs in grp
                fill!(llm, 0)
                xi = _x( obs)
                yi = _y( obs)
                vi = _nu(obs)

                # TODO use ProductionTempVars

                ψi = ones(1)

                plci = ProductionLikelihoodComputations(vi)
                llpsi = loglik_produce_scalars(pm, θ, plci)

                for m = 1:M
                    llm[m] = loglik_produce(llpsi..., ψi[1])
                end

                LL += logsumexp(llm)

                if dograd
                    softmax!(qm)
                    pgci = ProductionGradientComputations(qm, ψi, xi, vi)
                    grad_simloglik_produce!(grad, pm, θ, plci, pgci)
                end
            end
        end
        return LL - log(M)*num_i
    end

    gradtmp = zeros(Float64, length(pm,data))

    ff(θ::Vector) = -fg!(gradtmp, θ, y, X, false)
    function ffgg!(grad,θ)
        LL = -fg!(grad, θ, y, X, true)
        grad .*= -1
        return LL
    end

    # println("\n\nNelder Mead\n\n")
    res = optimize(ff, theta*2, NelderMead(), Optim.Options(time_limit = 5.0))
    @test maximum(abs.(res.minimizer[1:end-2] .- theta[1:end-2])) < 0.25
    @test maximum(abs.(res.minimizer[end-1:end].^2 .- theta[end-1:end].^2)) < 0.25

    # println("\n\nBFGS\n\n")
    od = OnceDifferentiable(ff, ffgg!, ffgg!, theta)
    res = optimize(ff, theta*1.1, BFGS(), Optim.Options(time_limit = 20.0))
    @test maximum(abs.(res.minimizer[1:end-2] .- theta[1:end-2])) < 0.2
    @test maximum(abs.(res.minimizer[end-1:end].^2 .- theta[end-1:end].^2)) < 0.2

    # check gradient
    fd = Calculus.gradient(ff, theta)
    fill!(gradtmp, 0)
    ffgg!(gradtmp, theta)
    @test norm(fd .- gradtmp, Inf) < 1e-5
end


@testset "Production full data" begin

    Random.seed!(1234)

    beta = rand(3)
    sigmas = (sqrt(0.25), sqrt(0.25), sqrt(0.1))
    theta = vcat(beta, sigmas...)
    data = DataProduce(10, 5, 5:10, theta)

end

# TODO
@testset "Production likelihood integration" begin
    @warn "Have not implemented this yet!!!"

    println("\n\n\t\tImplement test that gradient is all OK\n\n")
end


end # module
