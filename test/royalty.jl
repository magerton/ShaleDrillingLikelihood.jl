using ShaleDrillingLikelihood: RoyaltyModelNoHet,
    idx_roy_ρ, idx_roy_ψ, idx_roy_β, idx_roy_κ,
    theta_roy_ρ, theta_roy_ψ, theta_roy_β, theta_roy_κ,
    RoyaltyComputation,
    η12,
    lik_royalty, lik_loglik_royalty,
    theta_roy_check,
    llthreads!,
    loglik_royalty!,
    update_grad_royalty!

@testset "RoyaltyModelNoHet" begin
    k = 3
    nobs = 1_000
    L = 3
    RM = RoyaltyModelNoHet(L,k)

    X      = randn(k,nobs)
    eps    = randn(nobs)

    theta = [-2.0, 2.0, 2.0, -0.5, 0.5]  # β, κ

    @test length(theta) == length(RM)

    @test theta_roy_ρ(RM, theta) == Float64[]
    @test theta_roy_ψ(RM, theta) == Float64[]
    @test all(theta_roy_β(RM, theta) .== theta[1:3])
    @test theta_roy_κ(RM, theta, 1) == -0.5
    @test theta_roy_κ(RM, theta, 2) ==  0.5

    @test idx_roy_κ(RM, 1) == 4
    @test idx_roy_κ(RM, 2) == 5
    @test_throws DomainError idx_roy_κ(RM, L+1)
    @test_throws DomainError idx_roy_κ(RM, 0)

    rstar = X'*theta_roy_β(RM, theta) .+ eps
    l = map((r) ->  searchsortedfirst(theta_roy_κ(RM,theta), r), rstar)

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
    fg!(grad, theta)
    @test grad ≈ Calculus.gradient(f, theta)

    # check that it solves
    res = optimize(OnceDifferentiable(f, fg!, fg!, theta), theta*0.1, BFGS(), Optim.Options(time_limit = 1.0))
    @test maximum(abs.(res.minimizer .- theta)) < 0.21
end

@testset "RoyaltyModel" begin

    k = 3
    nobs = 1_000
    L = 3
    M = 10
    RM = RoyaltyModel(L,k)

    # make data
    X      = randn(k,nobs)
    eps    = randn(nobs)
    theta  = [0.0, 1.0, -2.0, 2.0, 2.0, -0.5, 0.5]  # β, κ

    # check indexing is correct
    @test all(theta_roy_β(RM, theta) .== theta[2 .+ (1:k)])
    @test all(theta_roy_κ(RM, theta) .== theta[end-L+2:end])

    # make more data
    rstar  = X'*theta_roy_β(RM, theta) .+ eps
    l = map((r) ->  searchsortedfirst(theta_roy_κ(RM,theta), r), rstar)

    # simulations
    u = randn(M,nobs)
    v = randn(M,nobs)
    Q = fill(Float64(1/M), M, nobs)

    A = similar(u)
    B = similar(u)
    C = similar(u)
    LL = similar(u)

    rc = RoyaltyComputation(l, X, A, B, C, LL, Q, u, v, 1)
    loglik_royalty!(rc, RM, theta, false)

    function f(θ::AbstractVector)
        lse = zeros(nobs)
        for i in 1:nobs
            rc = RoyaltyComputation(l, X, A, B, C, LL, Q, u, v, i)
            loglik_royalty!(rc, RM, θ, true)
            @views lse[i] = logsumexp(LL[:,i]) - log(M)
            @views softmax!(Q[:,i], LL[:,i])
        end
        return sum(lse)
    end

    function fg!(grad, θ)
        loglik = f(θ)
        @threads for i in 1:nobs
            rc = RoyaltyComputation(l, X, A, B, C, LL, Q, u, v, i)
            update_grad_royalty!(grad, RM, θ, rc)
        end
        return loglik
    end

    gradtmp = zeros(size(theta))

    fg!(gradtmp, theta)
    fd = Calculus.gradient(f, theta)
    @test gradtmp ≈ fd
end
