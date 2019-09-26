# ---------------------------------------------
# Tmp for liklihood
# ---------------------------------------------

"Tmp scalars for production liklihood"
struct ProductionLikelihoodComputations <: AbstractIntermediateComputations
    T::Int
    vpv::Float64
    vp1::Float64
end

function ProductionLikelihoodComputations(v::AbstractVector)
    T = length(v)
    vpv = dot(v,v)
    vp1 = sum(v)
    return ProductionLikelihoodComputations(T, vpv, vp1)
end

_T(    plc::ProductionLikelihoodComputations) = plc.T
_vpv(  plc::ProductionLikelihoodComputations) = plc.vpv
_vp1(  plc::ProductionLikelihoodComputations) = plc.vp1
_vp1sq(plc::ProductionLikelihoodComputations) = _vp1(plc)^2

# ---------------------------------------------
# Tmp for gradient
# ---------------------------------------------

"Tmp scalars + Vectors for production gradient"
struct ProductionGradientComputations <: AbstractIntermediateComputations
    ψbar::Float64
    ψ2bar::Float64
    Xpv::Vector{Float64}
    Xp1::Vector{Float64}
    function ProductionGradientComputations(ψbar::T, ψ2bar::T, Xpv::Vector{T}, Xp1::Vector{T}) where {T<:Float64}
        @assert length(Xpv)==length(Xp1)
        return new(ψbar, ψ2bar, Xpv, Xp1)
    end
end

_ψbar( pgc::ProductionGradientComputations) = pgc.ψbar
_ψ2bar(pgc::ProductionGradientComputations) = pgc.ψ2bar
_Xpv(  pgc::ProductionGradientComputations) = pgc.Xpv
_Xp1(  pgc::ProductionGradientComputations) = pgc.Xp1

function ψbar_ψ2bar(ψ, qm)
    ψbar  = dot(ψ, qm)
    ψ2bar = sumprod3(ψ, ψ, qm)
    return ψbar, ψ2bar
end

function ProductionGradientComputations(qm::AbstractVector,ψ::AbstractVector,X::AbstractMatrix,v::AbstractVector)
    psi_psi2 = ψbar_ψ2bar(ψ, qm)
    Xpv = X*v
    Xp1 = vec(sum(X; dims=2))
    return ProductionGradientComputations(psi_psi2..., Xpv, Xp1)
end

# ---------------------------------------------
# Production log lik
# ---------------------------------------------

function loglik_pdxn_scalars(model::ProductionModel, theta::AbstractVector, plc::ProductionLikelihoodComputations)

    αψ = theta_pdxn_ψ(  model, theta)
    σ2η = theta_pdxn_σ2η(model, theta)
    σ2u = theta_pdxn_σ2u(model, theta)
    a = σ2η^2
    b = σ2u^2

    T     = _T(    plc)
    vpv   = _vpv(  plc)
    vp1   = _vp1(  plc)
    vp1sq = _vp1sq(plc)

    abT = a + b*T
    ainv = 1/a
    c = b *ainv / abT
    ainv_cT = ainv - c*T

    A0 = -(T*log2π + (T-1)*log(a) + log(abT) + vpv*ainv - c*vp1sq) / 2
    A1 =    αψ*vp1*ainv_cT
    A2 = - (αψ^2*T*ainv_cT)/2

    return A0, A1, A2
end

loglik_pdxn(a::Real, b::Real, c::Real, ψ::Real) = a + b*ψ + c*ψ^2

# ---------------------------------------------
# Production gradient
# ---------------------------------------------

function dloglik_production!(grad::AbstractVector, model::ProductionModel, theta::AbstractVector, plc::ProductionLikelihoodComputations, pgc::ProductionGradientComputations)

    αψ = theta_pdxn_ψ( model, theta)
    σ2η = theta_pdxn_σ2η(model, theta)
    σ2u = theta_pdxn_σ2u(model, theta)
    a = σ2η^2
    b = σ2u^2

    ψbar  = _ψbar( pgc)
    ψ2bar = _ψ2bar(pgc)
    Xpv   = _Xpv(  pgc)
    Xp1   = _Xp1(  pgc)

    T     = _T(    plc)
    vpv   = _vpv(  plc)
    vp1   = _vp1(  plc)
    vp1sq = _vp1sq(plc)

    abT      = a + b*T
    abTinv   = 1/abT
    abTinvsq = abTinv^2
    ainv     = 1/a
    ainvsq   = ainv^2
    c        = b * ainv * abTinv
    ainv_cT  = ainv - c*T
    αψT      = αψ*T
    c_ainv_abTinv  = c*(ainv+abTinv)
    cT_ainv_abTinv = T*c_ainv_abTinv

    # ∂log L / ∂α_ψ
    grad[idx_pdxn_ψ(model)] += ainv_cT * (vp1*ψbar - αψ*T*ψ2bar)

    # ∂log L / ∂β
    H = (c*vp1 + αψ*ainv_cT*ψbar)
    grad[idx_pdxn_β(model)] .+= Xpv.*ainv .- H.*Xp1

    # ∂log L / ∂σ²η * ∂σ²η/∂ση
    E0 = -( (T-1)*ainv + abTinv - vpv*ainvsq + c_ainv_abTinv*vp1sq )/2
    E1 =   αψ*vp1*(-ainvsq + cT_ainv_abTinv)
    E2 = -(αψ^2*T*(-ainvsq + cT_ainv_abTinv))/2
    grad[idx_pdxn_σ2η(model)] += 2*σ2η*(E0 + E1*ψbar + E2*ψ2bar)

    # ∂log L / σ²u * * ∂σ²u/∂σu
    G0 = -(T*abTinv - vp1sq*abTinvsq)/2
    G1 = -αψT*vp1*abTinvsq
    G2 = ((αψT*abTinv)^2)/2
    grad[idx_pdxn_σ2u(model)] += 2*σ2u*(G0 + G1*ψbar + G2*ψ2bar)
end
