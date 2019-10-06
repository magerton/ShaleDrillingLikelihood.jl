module ShaleDrillingLikelihood_Production_Test

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using LinearAlgebra


using ShaleDrillingLikelihood: num_x,
    idx_produce_ψ, idx_produce_β, idx_produce_σ2η, idx_produce_σ2u,
    theta_produce, theta_produce_ψ, theta_produce_β, theta_produce_σ2η, theta_produce_σ2u,
    ProductionLikelihoodComputations, ProductionGradientComputations,
    grad_simloglik_produce!, loglik_produce,
    DataProduce, ObservationGroupProduce, ObservationProduce,
    _x, _y, _xsum, _nu, _i


@testset "Production basics" begin

    Random.seed!(1234)

    k = 3
    num_i = 500
    num_t = 11
    nobs = num_i * num_t
    pm = ProductionModel(k)
    M = 1

    X = randn(k,nobs)
    u = repeat(randn(num_i), inner=num_t)  # random effect
    η = randn(nobs)                        # idiosyncratic shock
    ψ = ones(M, num_i)
    theta = [1.0, 2.0, 3.0, 4.0, sqrt(0.10), sqrt(0.10)]

    @test length(theta) == length(pm)
    @test num_x(pm) == k

    @test theta_produce_ψ(pm, theta) == theta[1]
    @test all(theta_produce_β(pm, theta) .== theta[1 .+ (1:k)])
    @test theta_produce_σ2η(pm, theta) == theta[end-1]
    @test theta_produce_σ2u(pm, theta) == theta[end]

    v = sqrt(theta_produce_σ2u(pm,theta)^2) .* u .+ sqrt(theta_produce_σ2η(pm,theta)^2) .* η
    v .+= repeat(ψ[1,:], inner = num_t)
    y = v .+ X'*theta_produce_β(pm,theta)

    function viewi(v::AbstractVector, i::Integer)
        @assert 0 < i <= num_i
        @assert length(v) == num_t*num_i
        idx = (i-1)*num_t .+ (1:num_t)
        return view(v, idx)
    end

    function viewi(X::AbstractMatrix, i::Integer)
        @assert size(X) == (k, num_i*num_t,)
        @assert 0 < i <= num_i
        idx = (i-1)*num_t .+ (1:num_t)
        return view(X, :, idx)
    end

    xsum = Matrix{Float64}(undef, k, num_i)
    for i in 1:num_i
        sum!(xsum[:,i:i], viewi(X, i))
    end

    grpptr = collect(1:num_i+1)
    obsptr = collect(1:num_t:(num_i*num_t+1))

    data = DataProduce(y,X,xsum,zeros(nobs),obsptr, grpptr)

    function fg!(grad, θ, yy, xx, dograd::Bool=true)
        llm = zeros(Float64, M)
        qm = llm
        vi = zeros(Float64, num_t)

        # if any(θ[end-1:end] .<= 0)
        #     @warn "NEGATIVE σ^2η, σ^2_u = $(θ[end-1:end])"
        #     throw(error("NEGATIVE σ^2η, σ^2_u = $(θ[end-1:end])"))
        # end

        @assert length(llm) == size(ψ,1)

        _nu(data) .= _y(data) .- _x(data)'*theta_produce_β(pm, θ)

        LL = 0.0

        for grp in data
            for obs in grp
                fill!(llm, 0)
                xi = _x( obs)
                yi = _y( obs)
                vi = _nu(obs)

                # TODO use ProductionTempVars

                ψi = view(ψ, :, _i(grp))

                plci = ProductionLikelihoodComputations(vi)
                llpsi = ShaleDrillingLikelihood.loglik_produce_scalars(pm, θ, plci)

                for m = 1:M
                    llm[m] = loglik_produce(llpsi..., ψi[m])
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

    gradtmp = zeros(Float64, length(pm))

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
    res = optimize(ff, theta*1.5, BFGS(), Optim.Options(time_limit = 20.0))
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
    data = DataProduce(10, 5, 5:10, beta, sigmas)

end

# TODO
@testset "Production likelihood integration" begin
    @warn "Have not implemented this yet!!!"

    println("\n\n\t\tImplement test that gradient is all OK\n\n")
end


end # module
