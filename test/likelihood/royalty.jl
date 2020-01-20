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
using BenchmarkTools

using ShaleDrillingLikelihood: RoyaltyModelNoHet,
    idx_royalty_ρ, idx_royalty_ψ, idx_royalty_β, idx_royalty_κ,
    theta_royalty_ρ, theta_royalty_ψ, theta_royalty_β, theta_royalty_κ,
    η12,
    lik_royalty, lik_loglik_royalty,
    theta_royalty_check,
    simloglik_royalty!,
    grad_simloglik_royalty!,
    # _LLm,
    logsumexp_and_softmax!,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, num_choices, _num_x,
    SimulationDraws,
    update_ψ1!, update_dψ1dθρ!, _psi1, _u, _v, _dψ1dθρ,
    update_xbeta!, _am, _bm, _cm, _qm,
    llthreads!, _i, _nparm,
    logsumexp!,
    theta_royalty_level_to_cumsum,
    theta_royalty_cumsum_to_level,
    kappa_level_to_cumsum,
    kappa_cumsum_to_level


@testset "RoyaltyModelNoHet" begin
    k = 3
    nobs = 1_000
    L = 3
    RM = RoyaltyModelNoHet()
    Random.seed!(1234)

    X       = randn(k,nobs)
    epsilon = randn(nobs)

    theta0 = [-2.0, 2.0, 2.0, -0.5, 0.5]  # β, κ
    theta = vcat(theta0[1:end-1], sqrt(2*(theta0[end] - theta0[end-1])))
    @test kappa_level_to_cumsum(theta0[end-1:end]) ≈ theta[end-1:end]
    @test kappa_cumsum_to_level(theta[end-1:end]) ≈ theta0[end-1:end]
    @test length(theta) == k + L - 1

    rstar = X'*theta[1:k] .+ epsilon
    l = map((r) ->  searchsortedfirst(theta0[k+1:end], r), rstar)
    data = DataRoyalty(RM, l, X, 1:L)
    update_xbeta!(data, theta[1:k])

    @test theta ≈ theta_royalty_level_to_cumsum(data, theta0)
    @test theta0 ≈ theta_royalty_cumsum_to_level(data, theta)

    @test length(theta) == _nparm(data) == _nparm(first(first(data)))

    @test theta_royalty_ρ(    data, theta) == Float64[]
    @test theta_royalty_ψ(    data, theta) == Float64[]
    @test all(theta_royalty_β(data, theta) .== theta[1:3])
    @test theta_royalty_κ(    data, theta, 1) == theta[4]
    @test theta_royalty_κ(    data, theta, 2) == theta[5]

    @test idx_royalty_κ(data, 1) == 4
    @test idx_royalty_κ(data, 2) == 5
    @test_throws DomainError idx_royalty_κ(data, L+1)
    @test_throws DomainError idx_royalty_κ(data, 0)

    M = 10
    RT = SimulationDraws(M,nobs,0)
    am = _am(RT)
    bm = _bm(RT)
    cm = _cm(RT)

    # test η12s
    η12s = [η12(first(grp), theta, r) for (grp,r) in zip(data, rstar)]
    @test all(x[1] < x[2] for x in η12s )


    # form functions for stuff
    grad = similar(theta)
    # @code_warntype llthreads!(grad, theta, RM, data, true)

    f(parm) = - llthreads!(grad, parm, data, false)

    function fg!(g, parm)
        LL = llthreads!(g, parm, data, true)
        g .*= -1
        return -LL
    end

    # check gradient
    println("Print me to prevent error on v1.2")
    fg!(grad, theta)
    @test grad ≈ Calculus.gradient(f, theta)

    # check that it solves
    res = optimize(OnceDifferentiable(f, fg!, fg!, theta), theta*0.1, BFGS(), Optim.Options(time_limit = 1.0))
    @show res
    @show res.minimizer
    @test maximum(abs.(res.minimizer .- theta)) < 0.25
end

@testset "RoyaltyModel" begin

    nobs = 500  # observations
    k = 3       # number of x
    L = 3       # ordered choices
    M = 5_000     # simulations

    RM = RoyaltyModel()

    # make data
    X      = randn(k,nobs)
    eps    = randn(nobs)
    theta0  = [0.1, 1.0,    -2.0, 2.0, 2.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ
    theta = vcat(theta0[1:end-2], kappa_level_to_cumsum(theta0[end-1:end]))

    data = DataRoyalty(500, theta, L)

    # check indexing is correct
    @test theta_royalty_ρ(data, theta) == theta[1]
    @test theta_royalty_ψ(data, theta) == theta[2]
    @test all(theta_royalty_β(data, theta) .== theta[2 .+ (1:k)])
    @test all(theta_royalty_κ(data, theta) .== theta[end-L+2:end])

    # simulations
    uv = SimulationDraws(M,nobs)

    # let obs = first(first(data)), simi = view(uv,1)
    #     @code_warntype simloglik_royalty!(obs, theta, simi, false)
    # end

    let obs = first(first(data)), simi = view(uv,1)
        print("")
        # @show @benchmark simloglik_royalty!($obs, $theta, $simi, true)
        print("")
    end

    function fg!(grad::AbstractVector, θ::AbstractVector, dograd::Bool=true)
        LL = zero(eltype(θ))
        update_ψ1!(uv, theta_royalty_ρ(data,θ))
        update_dψ1dθρ!(uv, theta_royalty_ρ(data,θ))

        qm = _qm(uv)

        update_xbeta!(data, theta_royalty_β(data,θ))

        for grp in data
            for obs in grp
                fill!(qm, 0)                    # b/c might do other stuff to LLm
                simi = view(uv,_i(grp))
                simloglik_royalty!(obs, θ, simi, dograd)    # update LLm
                LL += logsumexp!(qm) - log(M)
                if dograd
                    grad_simloglik_royalty!(grad, obs, θ, simi)
                end
            end
        end
        return LL
    end


    gradtmp = zeros(size(theta))
    fg!(gradtmp, theta, false)
    fg!(gradtmp, theta, true)

    fd = Calculus.gradient(x -> fg!(gradtmp, x, false), theta)
    fill!(gradtmp, 0)
    fg!(gradtmp, theta, true)
    @test gradtmp ≈ fd
end

flush(stdout)

end # module
