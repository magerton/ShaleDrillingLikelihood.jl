using ShaleDrillingLikelihood: num_x,
    idx_pdxn_ψ, idx_pdxn_β, idx_pdxn_σ2η, idx_pdxn_σ2u,
    theta_pdxn, theta_pdxn_ψ, theta_pdxn_β, theta_pdxn_σ2η, theta_pdxn_σ2u,
    ProductionLikelihoodComputations, ProductionGradientComputations,
    dloglik_production!, loglik_pdxn


@testset "Production basics" begin

    k = 3
    num_i = 100
    num_t = 500
    nobs = num_i * num_t
    pm = ProductionModel(k)
    M = 1

    X = randn(k,nobs)
    u = repeat(randn(num_i), inner=num_t)  # random effect
    η = randn(nobs)                        # idiosyncratic shock
    ψ = ones(M, num_i)
    theta = [1.0, 2.0, 3.0, 4.0, 0.25, 0.5]

    @test length(theta) == length(pm)
    @test num_x(pm) == k

    @test theta_pdxn_ψ(pm, theta) == theta[1]
    @test all(theta_pdxn_β(pm, theta) .== theta[1 .+ (1:k)])
    @test theta_pdxn_σ2η(pm, theta) == theta[end-1]
    @test theta_pdxn_σ2u(pm, theta) == theta[end]

    v = sqrt(theta_pdxn_σ2u(pm,theta)) .* u .+ sqrt(theta_pdxn_σ2η(pm,theta)) .* η
    v .+= repeat(ψ[1,:], inner = num_t)
    y = v .+ X'*theta_pdxn_β(pm,theta)

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

    function fg!(grad, θ, yy, xx, dograd::Bool=true)
        llm = zeros(Float64, M)
        qm = llm
        vi = zeros(Float64, num_t)

        if any(θ[end-1:end] .<= 0)
            @warn "NEGATIVE σ^2η, σ^2_u = $(θ[end-1:end])"
            throw(error("NEGATIVE σ^2η, σ^2_u = $(θ[end-1:end])"))
        end

        @assert length(llm) == size(ψ,1)

        LL = 0.0
        for i in 1:num_i
            fill!(llm, 0)
            xi = viewi(xx, i)
            yi = viewi(yy, i)
            vi .= yi .- xi'*theta_pdxn_β(pm,θ)

            ψi = view(ψ, :, i)

            plci = ProductionLikelihoodComputations(vi)
            llpsi = ShaleDrillingLikelihood.loglik_pdxn_scalars(pm, θ, plci)

            for m = 1:M
                llm[m] = loglik_pdxn(llpsi..., ψi[m])
            end

            LL += logsumexp(llm)

            if dograd
                softmax!(qm)
                pgci = ProductionGradientComputations(qm, ψi, xi, vi)
                dloglik_production!(grad, pm, θ, plci, pgci)
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
    res = optimize(ff, theta*2, NelderMead(), Optim.Options(time_limit = 5.0))
    @test maximum(abs.(res.minimizer .- theta)) < 0.1

    fd = Calculus.gradient(ff, theta*2)
    fill!(gradtmp, 0)
    ffgg!(gradtmp, theta*2)
end
