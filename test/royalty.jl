module ShaleDrillingLikelihood_Royalty_Test

# using Revise

# using Juno
using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using InteractiveUtils

using ShaleDrillingLikelihood: RoyaltyModelNoHet,
    idx_royalty_ρ, idx_royalty_ψ, idx_royalty_β, idx_royalty_κ,
    theta_royalty_ρ, theta_royalty_ψ, theta_royalty_β, theta_royalty_κ,
    RoyaltyTmpVar,
    η12,
    lik_royalty, lik_loglik_royalty,
    theta_royalty_check,
    simloglik_royalty!,
    grad_simloglik_royalty!,
    _LLm,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, _num_choices, num_x,
    SimulationDraws, RoyaltyLikelihoodInformation,
    update_ψ1!, update_dψ1dρ!, _psi1, _u, _v, _dψ1dρ,
    update_xbeta!, _am, _bm, _cm,
    llthreads!

@testset "RoyaltyModelNoHet" begin
    k = 3
    nobs = 1_000
    L = 3
    RM = RoyaltyModelNoHet()
    Random.seed!(1234)

    X      = randn(k,nobs)
    eps    = randn(nobs)

    theta = [-2.0, 2.0, 2.0, -0.5, 0.5]  # β, κ
    @assert length(theta) == k + L - 1

    rstar = X'*theta[1:k] .+ eps
    l = map((r) ->  searchsortedfirst(theta[k+1:end], r), rstar)
    data = DataRoyalty(l, X)
    update_xbeta!(data, theta[1:k])

    @test length(theta) == length(RM, data)

    @test theta_royalty_ρ(    RM, data, theta) == Float64[]
    @test theta_royalty_ψ(    RM, data, theta) == Float64[]
    @test all(theta_royalty_β(RM, data, theta) .== theta[1:3])
    @test theta_royalty_κ(    RM, data, theta, 1) == -0.5
    @test theta_royalty_κ(    RM, data, theta, 2) ==  0.5

    @test idx_royalty_κ(RM, data, 1) == 4
    @test idx_royalty_κ(RM, data, 2) == 5
    @test_throws DomainError idx_royalty_κ(RM, data, L+1)
    @test_throws DomainError idx_royalty_κ(RM, data, 0)

    M = 10
    RT = RoyaltyTmpVar(M)
    am = _am(RT)
    bm = _bm(RT)
    cm = _cm(RT)

    # test η12s
    η12s = [η12(obs, RM, theta, r) for (obs,r) in zip(data, rstar)]
    @test all(x[1] < x[2] for x in η12s )


    # form functions for stuff
    grad = similar(theta)
    # @code_warntype llthreads!(grad, theta, RM, data, true)

    f(parm) = - llthreads!(grad, parm, RM, data, false)

    function fg!(g, parm)
        LL = llthreads!(g, parm, RM, data, true)
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

    RM = RoyaltyModel()

    # make data
    X      = randn(k,nobs)
    eps    = randn(nobs)
    theta  = [0.1, 1.0,    -2.0, 2.0, 2.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ

    # make more data
    rstar  = X'*theta[2 .+ (1:k)] .+ eps
    l = map((r) ->  searchsortedfirst(theta[end-L+2:end], r), rstar)
    data = DataRoyalty(l,X)

    # check indexing is correct
    @test all(theta_royalty_β(RM, data, theta) .== theta[2 .+ (1:k)])
    @test all(theta_royalty_κ(RM, data, theta) .== theta[end-L+2:end])

    # simulations
    uv = SimulationDraws(M,nobs)
    rc = RoyaltyTmpVar(M)
    rli = RoyaltyLikelihoodInformation(rc, data[1], RM, view(uv,1))

    # @code_warntype simloglik_royalty!(rli, theta, false)

    function fg!(grad::AbstractVector, θ::AbstractVector, dograd::Bool=true)
        LL = zero(eltype(θ))
        update_ψ1!(uv, theta_royalty_ρ(RM,data,θ))
        update_dψ1dρ!(uv, theta_royalty_ρ(RM,data,θ))
        qm = ShaleDrillingLikelihood._qm(rc)
        update_xbeta!(data, theta_royalty_β(RM,data,θ))
        for i in 1:nobs
            fill!(_LLm(rc), 0)                    # b/c might do other stuff to LLm
            rli = RoyaltyLikelihoodInformation(rc, data[i], RM, view(uv,i))
            simloglik_royalty!(rli, θ, dograd)    # update LLm
            LL += logsumexp(_LLm(rc))             # b/c integrating
            softmax!(qm, _LLm(rc))                # b/c grad needs qm = Pr(m|data)
            dograd && grad_simloglik_royalty!(grad, rli, θ)
        end
        return LL - nobs*log(M)
    end


    gradtmp = zeros(size(theta))
    fg!(gradtmp, theta, false)
    fg!(gradtmp, theta, true)

    fd = Calculus.gradient(x -> fg!(gradtmp, x, false), theta)
    fill!(gradtmp, 0)
    fg!(gradtmp, theta, true)
    @test gradtmp ≈ fd
end


end # module
