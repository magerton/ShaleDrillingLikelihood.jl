module ShaleDrillingLikelihood_Royalty_Test

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
# using InteractiveUtils

using ShaleDrillingLikelihood: RoyaltyModelNoHet,
    idx_royalty_ρ, idx_royalty_ψ, idx_royalty_β, idx_royalty_κ,
    theta_royalty_ρ, theta_royalty_ψ, theta_royalty_β, theta_royalty_κ,
    RoyaltyComputation,
    η12,
    lik_royalty, lik_loglik_royalty,
    theta_royalty_check,
    simloglik_royalty!,
    grad_simloglik_royalty!,
    _LLm,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, _num_choices, num_x

@testset "RoyaltyModelNoHet" begin
    k = 3
    nobs = 1_000
    L = 3
    RM = RoyaltyModelNoHet(L,k)
    Random.seed!(1234)

    X      = randn(k,nobs)
    eps    = randn(nobs)

    theta = [-2.0, 2.0, 2.0, -0.5, 0.5]  # β, κ

    @test length(theta) == length(RM)

    @test theta_royalty_ρ(RM, theta) == Float64[]
    @test theta_royalty_ψ(RM, theta) == Float64[]
    @test all(theta_royalty_β(RM, theta) .== theta[1:3])
    @test theta_royalty_κ(RM, theta, 1) == -0.5
    @test theta_royalty_κ(RM, theta, 2) ==  0.5

    @test idx_royalty_κ(RM, 1) == 4
    @test idx_royalty_κ(RM, 2) == 5
    @test_throws DomainError idx_royalty_κ(RM, L+1)
    @test_throws DomainError idx_royalty_κ(RM, 0)

    rstar = X'*theta_royalty_β(RM, theta) .+ eps
    l = map((r) ->  searchsortedfirst(theta_royalty_κ(RM,theta), r), rstar)

    data = DataRoyalty(l, X)

    M = 10
    am = Vector{Float64}(undef,M)
    bm = similar(am)
    cm = similar(bm)

    η12s = map((lr) -> η12(RM, theta, lr[1], lr[2]), zip(l, rstar))
    @test all(map((x) -> x[1] < x[2], η12s))

    # @code_warntype ShaleDrillingLikelihood.llthreads!(grad, theta, RM, l, X, true)

    # form functions for stuff
    grad = similar(theta)
    f(parm) = - ShaleDrillingLikelihood.llthreads!(grad, parm, RM, l, X, false)

    function fg!(g, parm)
        LL = ShaleDrillingLikelihood.llthreads!(g, parm, RM, l, X, true)
        g .*= -1
        return -LL
    end

    # check gradient
    println("Print me to prevent error on v1.2")
    fg!(grad, theta)
    @test grad ≈ Calculus.gradient(f, theta)

    # check that it solves
    res = optimize(OnceDifferentiable(f, fg!, fg!, theta), theta*0.1, BFGS(), Optim.Options(time_limit = 1.0))
    @test maximum(abs.(res.minimizer .- theta)) < 0.25
end

@testset "RoyaltyModel" begin

    nobs = 500  # observations
    k = 3       # number of x
    L = 3       # ordered choices
    M = 15      # simulations

    RM = RoyaltyModel(L,k)

    # make data
    X      = randn(k,nobs)
    eps    = randn(nobs)
    theta  = [0.0, 1.0,    -2.0, 2.0, 2.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ

    # check indexing is correct
    @test all(theta_royalty_β(RM, theta) .== theta[2 .+ (1:k)])
    @test all(theta_royalty_κ(RM, theta) .== theta[end-L+2:end])

    # make more data
    rstar  = X'*theta_royalty_β(RM, theta) .+ eps
    l = map((r) ->  searchsortedfirst(theta_royalty_κ(RM,theta), r), rstar)

    # simulations
    u = randn(M,nobs)
    v = randn(M,nobs)

    qm = fill(Float64(1/M), M)
    am  = similar(qm)
    bm  = similar(qm)
    cm  = similar(qm)
    llm = similar(qm)

    rc = RoyaltyComputation(l, X, am, bm, cm, llm, qm, u, v, 1)
    # @code_warntype simloglik_royalty!(rc, RM, theta, false)

    function fg!(grad::AbstractVector, θ::AbstractVector, dograd::Bool=true)
        LL = zero(eltype(θ))
        for i in 1:nobs
            rc = RoyaltyComputation(l, X, am, bm, cm, llm, qm, u, v, i)
            fill!(_LLm(rc), 0)                    # b/c might do other stuff to LLm
            simloglik_royalty!(rc, RM, θ, dograd)    # update LLm
            LL += logsumexp(_LLm(rc))             # b/c integrating
            softmax!(qm, _LLm(rc))                # b/c grad needs qm = Pr(m|data)
            dograd && grad_simloglik_royalty!(grad, RM, θ, rc)
        end
        return LL - nobs*log(M)
    end

    gradtmp = zeros(size(theta))
    fd = Calculus.gradient(x -> fg!(gradtmp, x, false), theta)
    fg!(gradtmp, theta, true)
    @test gradtmp ≈ fd
end


end # module
