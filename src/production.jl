
"Tmp scalars for production computations"
struct ProductionLikelihoodComputations{V2<:AbstractVector{Float64}, V1<:AbstractVector{Float64}, M<:AbstractMatrix{Float64}} <: AbstractIntermediateComputations
    T::Int
    @addStructFields(Float64, a, b, c, αψ, vpv, vp1)
end

function ψbar_ψ2bar(ψ, qm)
    ψbar  = dot(ψ, qm)
    ψ2bar = sumprod3(ψ, ψ, qm)
    return ψbar, ψ2bar
end

function ProductionLikelihoodComputations(tempvars, data, model, theta)
    T = length(y)

    a = theta_pdxn_σ2η(model, theta)
    b = theta_pdxn_σ2u(model, theta)
    αψ = theta_pdxn_ψ(model, theta)
    v = view(_v(tempvars), 1:T)

    vpv = dot(v,v)
    vp1 = sum(v)

    return ProductionComputations(T, a, b, αψ, vpv, vp1)
end


_nobs(   pc::ProductionLikelihoodComputations) = pc.T
_a(      pc::ProductionLikelihoodComputations) = pc.a
_b(      pc::ProductionLikelihoodComputations) = pc.b
_c(      pc::ProductionLikelihoodComputations) = nothing # FIXME
_vpv(    pc::ProductionLikelihoodComputations) = pc.vpv
_vp1(    pc::ProductionLikelihoodComputations) = pc.vp1
_αψ(     pc::ProductionLikelihoodComputations) = pc.αψ

_vp1sq(  pc::ProductionLikelihoodComputations) = vp1(pc)^2
_Xpv(    pc::ProductionLikelihoodComputations) = pc.Xpv

# ---------------------------------------------
# Production log lik
# ---------------------------------------------

function loglik_pdxn_scalars(model::ProductionModel, theta::AbstractVector, v::AbstractVector)

    T = qlength(pdxn_parm)

    a = theta_pdxn_σ2η(model, theta)
    b = theta_pdxn_σ2u(model, theta)
    α = theta_pdxn_ψ(model, theta)

    abT = a + b*T
    c = b / (a * abT)
    cT = c*T
    ainv = 1/a

    vpv = dot(v,v)
    vp1 = sum(v)
    vp1sq = vp1^2

    A0 = (-T*log2π - (T-1)*log(a) - log(abT) - vpv/a + c*vp1sq) / 2
    A1 =  α*vp1*(ainv + cT)
    A2 = (α^2*T*(ainv - cT))/2

    return A0, A1, A2
end

loglik_pdxn(a::Real, b::Real, c::Real, ψ::Real) = a + b*ψ + c*ψ^2

loglik_pdxn(model, theta, v, ψ::Number) = loglik_pdxn(loglik_pdxn_scalars(model, theta, v)..., ψ)

# ---------------------------------------------
# Production gradient
# ---------------------------------------------

function dloglik_production!(grad::AbstractVector, model)

    T = qlength(pdxn_parm)

    α = theta_pdxn_ψ(  model, theta)
    a = theta_pdxn_σ2η(model, theta)
    b = theta_pdxn_σ2u(model, theta)

    bT = b*T
    abT = a + bT
    c = b / (a * abT)
    αT = α*T
    cT = c*T
    ainv = 1/a
    binv = 1/b
    abTinv = 1/abT

    vpv = dot(v,v)
    vp1 = sum(v)
    vp1sq = vp1^2

    update_Xpv!(tmp_pdxn, data)
    update_Xp1!(tmp_pdxn, data)
    ψbar, ψ2bar = ψbar_ψ2bar(ψ, qm)

    # ∂log L / ∂α_ψ
    B = (ainv - cT)
    grad[idx_pdxn_ψ(model)] += vp1*B*ψbar - α*T*B*ψ2bar

    # ∂log L / ∂β
    H = (c*vp1 + α*B*ψbar)
    @. grad[idx_pdxn_β(model)] += Xpv*ainv - H*Xp1

    # ∂log L / ∂σ²η
    E0 = -((T-1)*ainv + abTinv)
    E1 =  ainv^2
    E2 = -c * (ainv - abTinv)
    E1TE2 = (E1 + T*E2)

    grad[idx_pdxn_σ2η(model)] += 0.5 * ( (E0 + E1*vpv + E2*vp1sq) - 2*α*vp1*E1TE2*ψbar + α^2*T*E1TE2*ψ2bar )

    # ∂log L / σ²u
    G0 = -T * abTinv
    G1 =  c * ( binv - T*abTinv )
    grad[idx_pdxn_σ2u(model)] += 0.5 * ( (G0 + G1*vp1sq) - αT*G1*(2*vp1*ψbar + αT*ψ2bar) )
end
