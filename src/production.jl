# ---------------------------------------------
# Production log lik
# ---------------------------------------------

function loglik_produce_scalars(obs::ObservationProduce, model::ProductionModel, theta::AbstractVector)

    αψ  = theta_produce_ψ(  model, obs, theta)
    σ2η = theta_produce_σ2η(model, obs, theta)
    σ2u = theta_produce_σ2u(model, obs, theta)
    a = σ2η^2
    b = σ2u^2

    T     = length(obs)
    vpv   = _nusumsq(obs)
    vp1   = _nusum(  obs)
    vp1sq = vp1^2

    abT = a + b*T
    ainv = 1/a
    c = b *ainv / abT
    ainv_cT = ainv - c*T

    A0 = -(T*log2π + (T-1)*log(a) + log(abT) + vpv*ainv - c*vp1sq) / 2
    A1 =    αψ*vp1*ainv_cT
    A2 = - (αψ^2*T*ainv_cT)/2

    return A0, A1, A2
end

loglik_produce(a::Real, b::Real, c::Real, ψ::Real) = a + (b + c*ψ)*ψ

function simloglik_produce!(obs::ObservationProduce, model::ProductionModel, theta::AbstractVector, sim::SimulationDrawsVector)
    a, b, c = loglik_produce_scalars(obs, model, theta)
    f(x) = loglik_produce(a,b,c,x)
    _qm(sim) .+= f.(_psi2(sim))
end

# ---------------------------------------------
# Production gradient
# ---------------------------------------------

function grad_simloglik_produce!(grad::AbstractVector, obs::ObservationProduce, model::ProductionModel, theta::AbstractVector, sim::SimulationDrawsVector)

    αψ  = theta_produce_ψ( model, theta)
    σ2η = theta_produce_σ2η(model, theta)
    σ2u = theta_produce_σ2u(model, theta)
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
    grad[idx_produce_ψ(model,obs)] += ainv_cT * (vp1*ψbar - αψT*ψ2bar)

    # ∂log L / ∂β
    H = c*vp1 + αψ*ainv_cT*ψbar
    grad[idx_produce_β(model,obs)] .+= Xpv.*ainv .- H.*Xp1

    # ∂log L / ∂σ²η * ∂σ²η/∂ση
    E0 = -( (T-1)*ainv + abTinv - vpv*ainvsq + c_ainv_abTinv*vp1sq )/2
    Einner = αψ*(-ainvsq + T*c_ainv_abTinv)
    E1 =   Einner*vp1
    E2 = -(αψ*T*Einner)/2
    grad[idx_produce_σ2η(model,obs)] += 2*σ2η*(E0 + E1*ψbar + E2*ψ2bar)

    # ∂log L / σ²u * * ∂σ²u/∂σu
    G0 = -(T*abTinv - vp1sq*abTinvsq)/2
    G1 = -αψT*vp1*abTinvsq
    G2 = ((αψT*abTinv)^2)/2
    grad[idx_produce_σ2u(model,obs)] += 2*σ2u*(G0 + G1*ψbar + G2*ψ2bar)
end
