# ---------------------------------------------
# Production log lik
# ---------------------------------------------

"Production"
struct ProductionModel <: AbstractModel
    num_x::Int
end

num_x(m::ProductionModel) = m.num_x

idx_pdxn_αψ( m::ProductionModel) = 1
idx_pdxn_β(  m::ProductionModel) = 1 .+ (1:num_x(m))
idx_pdxn_σ2η(m::ProductionModel) = 2 + num_x(m)
idx_pdxn_σ2u(m::ProductionModel) = 3 + num_x(m)

theta_pdxn(m::ProductionModel, theta) = theta

function abc_loglik_pdxn(pdxn_parm)

    T = _nobs(pdxn_parm)
    a = _a(pdxn_parm)
    b = _b(pdxn_parm)
    c = _c(pdxn_parm)
    vpv = sumsq_nu(pdxn_parm)
    vp1 = sum_nu(pdxn_parm)
    vp1sq = sum_nusq(pdxn_parm)
    α = alpha(pdxn_parm)

    A0 = (-T*log2π - (T-1)*log(a) - log(a+bT) - vpv/a + c*vp1sq) / 2
    A1 = α*vp1*(1/a + c*T)
    A2 = (α^2*T*(1/a - c*T))/2

    return A0, A1, A2
end

loglik_pdxn(psi1, a, b, c) = a + b*psi1 + b*psi1^2

# ---------------------------------------------
# Production gradient
# ---------------------------------------------


struct EstimationTempVars
    v::Vector{Float64}
    Xpv::Vector{Float64}
end

_v(e::EstimationTempVars) = e.nu
_Xpv(e::EstimationTempVars) = e.Xpv

"Tmp scalars for production computations"
struct ProductionComputations{V2<:AbstractVector{Float64}, V1<:AbstractVector{Float64}, M<:AbstractMatrix{Float64}} <: AbstractIntermediateComputations
    T::Int
    @addStructFields(Float64, a, b, c, vpv, vp1, α)
    Xpv::Vector{Float64}
    v::V1
    Xp1::V2
    X::M
end

function ψbar_ψ2bar(ψ1, qm)
    # https://github.com/jw3126/ThreadingTools.jl
    # https://discourse.julialang.org/t/how-to-speed-up-this-simple-code-multithreading-simd-inbounds/19681/39
    # https://discourse.julialang.org/t/parallelizing-for-loop-in-the-computation-of-a-gradient/9154/7?u=tkoolen
    # https://discourse.julialang.org/t/innefficient-paralellization-need-some-help-optimizing-a-simple-dot-product/9723/31
    # https://discourse.julialang.org/t/for-loops-acceleration/24011/7
    ψbar  = sum(ψ1 .* qm)    # TODO threaded
    ψ2bar = sum(ψ1.^2 .* qm) # TODO threaded
    return ψbar, ψ2bar
end

function ProductionComputations(tempvars, data, model, theta)
    T = length(y)

    α = theta_αψ(      model, theta)
    β = theta_pdxn_β(   model, theta)
    a = theta_pdxn_σ2η(model, theta)
    b = theta_pdxn_σ2u(model, theta)

    v = view(_v(tempvars), 1:T)
    v .= y
    BLAS.gemv!('T', -1.0, X, β, 1.0, v)

    # TODO threaded
    vpv = sum(v.^2)
    vp1 = sum(v)

    return ProductionComputations(T, a, b, c, vpv, vp1, α, _Xpv(tempvars), v, _Xsum(data, i, w))
end

update_Xpv!(tmp_pdxn) = mul!(Xpv(tmp_pdxn), X(tmp_pdxn), v(tmp_pdxn))

_nobs(   pc::ProductionComputations) = pc.T
_a(      pc::ProductionComputations) = pc.a
_b(      pc::ProductionComputations) = pc.b
_c(      pc::ProductionComputations) = pc.c
_alpha(  pc::ProductionComputations) = pc.α
_vpv(    pc::ProductionComputations) = pc.vpv
_vp1(    pc::ProductionComputations) = pc.vp1
_vp1sq(  pc::ProductionComputations) = vp1(pc)^2
_Xpv(    pc::ProductionComputations) = pc.Xpv
_v(      pc::ProductionComputations) = pc.v


function dloglik_production!(grad::AbstractVector, model, pdxn_parm::ProductionComputations, pdxn_grad_parm::ProductionGradientComputations, ψbar, ψ2bar)

    T     = _nobs(pdxn_parm)
    a     = _a(pdxn_parm)
    b     = _b(pdxn_parm)
    c     = _c(pdxn_parm)
    α     = alpha(pdxn_parm)
    vpv   = _vpv(  pdxn_parm)
    vp1   = _vp1(  pdxn_parm)
    vp1sq = _vp1sq(pdxn_parm)

    update_Xpv!(tmp_pdxn)

    bT = b*T
    αT = α*T
    cT = c*T
    ainv = 1/a
    binv = 1/b
    abT = a + bT
    abTinv = 1/abT

    # ∂log L / ∂α_ψ
    B = (ainv - cT)
    grad[idx_αψ(model)] += vp1*B*ψbar - α*T*B*ψ2bar

    # ∂log L / ∂β
    H = (c*vp1 + α*B*ψbar)
    @. grad[idx_βpdxn(model)] += Xpv*ainv - H*Xp1

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
