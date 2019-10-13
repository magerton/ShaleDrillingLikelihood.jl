# ---------------------------------------------
# Production log lik
# ---------------------------------------------

function loglik_produce_scalars(obs::ObservationProduce, model::ProductionModel, theta::AbstractVector)

    αψ  = theta_produce_ψ(  obs, theta)
    σ2η = theta_produce_σ2η(obs, theta)
    σ2u = theta_produce_σ2u(obs, theta)
    a = σ2η^2
    b = σ2u^2

    T   = length(obs)
    vpv = _nusumsq(obs)
    vp1 = _nusum(  obs)

    abT = a + b*T
    ainv = 1/a
    c = b *ainv / abT
    ainv_cT = ainv - c*T

    A0 = -(T*log2π + (T-1)*log(a) + log(abT) + vpv*ainv - c*vp1^2) / 2
    A1 =    αψ*vp1*ainv_cT
    A2 = - (αψ^2*T*ainv_cT)/2

    return A0, A1, A2
end

function simloglik_produce!(obs::ObservationProduce, model::ProductionModel, theta::AbstractVector, sim::SimulationDrawsVector)
    a, b, c = loglik_produce_scalars(obs, model, theta)
    f(x) = a + (b + c*x)*x
    qm = _qm(sim) # breaks 1.1.0
    qm .+= f.(_psi2(sim))
end

# ---------------------------------------------
# Production gradient
# ---------------------------------------------

function grad_simloglik_produce!(grad::AbstractVector, obs::ObservationProduce, model::ProductionModel, theta::AbstractVector, sim::SimulationDrawsVector)

    αψ  = theta_produce_ψ(  obs, theta)
    σ2η = theta_produce_σ2η(obs, theta)
    σ2u = theta_produce_σ2u(obs, theta)
    a = σ2η^2
    b = σ2u^2

    ψbar, ψ2bar = psi2_wtd_sum_and_sumsq(sim)

    Xpv   = _xpnu(obs)
    Xp1   = _xsum(obs)
    T     = length(obs)
    vpv   = _nusumsq(obs)
    vp1   = _nusum(obs)
    vp1sq = vp1^2

    abT      = a + b*T
    abTinv   = 1/abT
    abTinvsq = abTinv^2
    ainv     = 1/a
    ainvsq   = ainv^2
    c        = b * ainv * abTinv
    ainv_cT  = ainv - c*T
    αψT      = αψ*T
    c_ainv_abTinv  = c*(ainv+abTinv)

    # ∂log L / ∂α_ψ
    grad[idx_produce_ψ(obs)] += ainv_cT * (vp1*ψbar - αψT*ψ2bar)

    # ∂log L / ∂β
    H = c*vp1 + αψ*ainv_cT*ψbar
    grad[idx_produce_β(obs)] .+= Xpv.*ainv .- H.*Xp1

    # ∂log L / ∂σ²η * ∂σ²η/∂ση
    E0 = -( (T-1)*ainv + abTinv - vpv*ainvsq + c_ainv_abTinv*vp1sq )/2
    Einner = αψ*(-ainvsq + T*c_ainv_abTinv)
    E1 =   Einner*vp1
    E2 = -(αψT*Einner)/2
    grad[idx_produce_σ2η(obs)] += 2*σ2η*(E0 + E1*ψbar + E2*ψ2bar)

    # ∂log L / σ²u * * ∂σ²u/∂σu
    G0 = -(T*abTinv - vp1sq*abTinvsq)/2
    G1 = -αψT*vp1*abTinvsq
    G2 = ((αψT*abTinv)^2)/2
    grad[idx_produce_σ2u(obs)] += 2*σ2u*(G0 + G1*ψbar + G2*ψ2bar)
end



function grad_simloglik_produce!(
    grad::AbstractVector, data::DataProduce, model::ProductionModel,
    θ::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
)
    qm = _qm(sim)
    M = _num_sim(sim)

    # given θ update ν = y - X'β
    update_nu!(data, θ)
    update_xpnu!(data)
    fill!(grad, 0)

    LL = 0.0
    for (i,grp) in enumerate(data)
        simi = view(sim, i)
        fill!(qm, 0)

        for obs in grp
            simloglik_produce!(obs, model, θ, simi)
        end

        if !dograd
            LL += logsumexp(qm) - log(M)
        else
            LL += logsumexp_and_softmax!(qm) - log(M)
            for obs in grp
                grad_simloglik_produce!(grad, obs, model, θ, simi)
            end
        end
    end
    return LL
end
