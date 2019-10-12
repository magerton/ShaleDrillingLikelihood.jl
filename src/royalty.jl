using StatsFuns
using Distributions: _F1
using Base.Threads

# ---------------------------------------------
# Royalty computations for individual `i`
# ---------------------------------------------

"Tempvars for royalty simulations with `i`"
struct RoyaltyTmpVar{T<:Real} <: AbstractIntermediateComputations
    am::Vector{T}
    bm::Vector{T}
    cm::Vector{T}
    LLm::Vector{T}
    qm::Vector{T}

    function RoyaltyTmpVar(am::V, bm::V, cm::V, LLm::V, qm::V) where {T,V<:Vector{T}}
        length(am)==length(bm)==length(cm)==length(qm)==length(LLm) || throw(DimensionMismatch())
        return new{T}(am, bm, cm, LLm, qm)
    end
end

_am( rc::RoyaltyTmpVar) = rc.am
_bm( rc::RoyaltyTmpVar) = rc.bm
_cm( rc::RoyaltyTmpVar) = rc.cm
_LLm(rc::RoyaltyTmpVar) = rc.LLm
_qm( rc::RoyaltyTmpVar) = rc.qm


function RoyaltyTmpVar(M)
    am = Vector{Float64}(undef,M)
    bm = similar(am)
    cm = similar(am)
    llm = similar(am)
    qm = similar(am)
    RoyaltyTmpVar(am, bm, cm, llm, qm)
end


"What we need to compure royalty"
struct RoyaltyLikelihoodInformation{RT<:RoyaltyTmpVar, O<:ObservationRoyalty, M<:AbstractRoyaltyModel, S<:SimulationDrawsVector}
    tmpvar::RT
    observation::O
    model::M
    simulated_draws::S
end

_tmpvar(rli::RoyaltyLikelihoodInformation) = rli.tmpvar
_observation(rli::RoyaltyLikelihoodInformation) = rli.observation
_model(rli::RoyaltyLikelihoodInformation) = rli.model
_draws(rli::RoyaltyLikelihoodInformation) = rli.simulated_draws

# ---------------------------------------------
# Royalty low level computations
# ---------------------------------------------

function η12(obs::ObservationRoyalty, model::AbstractRoyaltyModel, theta::AbstractVector, zm::Real)
    l = _y(obs)
    L = _num_choices(obs)
    η1 = l == 1 ? typemin(zm) : theta_royalty_κ(model, obs, theta, l-1) - zm
    η2 = l == L ? typemax(zm) : theta_royalty_κ(model, obs, theta, l)   - zm
    @assert η1 < η2
    return η1, η2
end

const invsqrthalfπ = 1/sqrthalfπ

function dlogcdf_trunc(a::Real, b::Real)
    # https://github.com/scipy/scipy/blob/a2ffe09aa751749f2372aa13c19c61b2dec5266f/scipy/stats/_continuous_distns.py
    # https://github.com/JuliaStats/Distributions.jl/blob/master/src/truncated/normal.jl
    # https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf
    a == typemin(a) && return   normpdf(b) / normcdf(b)
    b == typemax(b) && return - normpdf(a) / normccdf(a)
    return - _F1(a*invsqrt2, b*invsqrt2) * invsqrthalfπ
end


function lik_royalty(obs::ObservationRoyalty, η12::NTuple{2,Real})
    η1, η2 = η12
    l = _y(obs)
    if l == 1
        return normcdf(η2)
    elseif l == _num_choices(obs)
        return normccdf(η1)
    else
        return normcdf(η2) - normcdf(η1)
    end
end


function loglik_royalty(obs::ObservationRoyalty, η12::NTuple{2,Real})
    η1, η2 = η12
    l = _y(obs)
    if l == 1
        return normlogcdf(η2)
    elseif l == _num_choices(obs)
        return normlogccdf(η1)
    else
        return log(normcdf(η2) - normcdf(η1))
    end
    return F
end


function lik_loglik_royalty(obs::ObservationRoyalty, η12::NTuple{2,Real})
    l = _y(obs)
    F = lik_royalty(obs, η12)
    η1, η2 = η12
    if l == 1
        return  F, normlogcdf(η2)
    elseif l == _num_choices(obs)
        return  F, normlogccdf(η1)
    else
        return F, log(F)
    end
end

# ---------------------------------------------
# Royalty likelihood
# ---------------------------------------------

"this is the main function. for each ψm, it computes LLm(roy|ψm) and ∇LLm(roy|ψm)"
function simloglik_royalty!(rli::RoyaltyLikelihoodInformation, theta::AbstractVector, dograd::Bool)

    rc    = _tmpvar(rli)
    obs   = _observation(rli)
    model = _model(rli)
    draws = _draws(rli)

    am = _am(rc)
    bm = _bm(rc)
    cm = _cm(rc)
    LLm = _LLm(rc)

    u = _u(draws)
    v = _v(draws)
    psi = _ψ1(draws)

    l = _y(obs)
    xbeta = _xbeta(obs)

    ρ  = theta_royalty_ρ(model, obs, theta)
    βψ = theta_royalty_ψ(model, obs, theta)
    βx = theta_royalty_β(model, obs, theta)

    # @inbounds @threads
    for m = 1:length(u)
        zm  = xbeta + βψ * psi[m]
        eta12 = η12(obs, model, theta, zm)

        if dograd == false
            LLm[m] += loglik_royalty(obs, eta12)
        else
            η1, η2 = eta12
            F,LL  = lik_loglik_royalty(obs, eta12)
            LLm[m] += LL
            am[m] = normpdf(η1) / F
            bm[m] = normpdf(η2) / F
            cm[m] = dlogcdf_trunc(η1, η2)
        end
    end
    return nothing
end

# ---------------------------------------------
# Royalty gradient
# ---------------------------------------------

"gradient for simulated likelihood when integrating"
function grad_simloglik_royalty!(grad::AbstractVector, rli::RoyaltyLikelihoodInformation, theta::AbstractVector)

    rc    = _tmpvar(rli)
    obs   = _observation(rli)
    model = _model(rli)
    draws = _draws(rli)

    @assert length(grad) == length(theta) == length(model,obs)

    am = _am(rc) # computed in likelihood
    bm = _bm(rc) # computed in likelihood
    cm = _cm(rc) # computed in likelihood
    qm = _qm(rc) # Pr(um,vm | data) already computed!!!

    l = _y(obs)         # discrete choice info
    L = _num_choices(obs)  # discrete choice info
    x = _x(obs)

    ρ  = theta_royalty_ρ(model, obs, theta) # parameters
    βψ = theta_royalty_ψ(model, obs, theta) # parameters

    ψ1 = _ψ1(draws)
    dψ1dρ = _dψ1dρ(draws)

    # gradient
    grad[idx_royalty_ρ(model,obs)] -= sumprod3(dψ1dρ, qm, cm) * βψ   # tmapreduce((q,c,u,v) -> q*c*dψdρ(u,v), +, qm,cm,um,vm; init=zero(eltype(qm))) * βψ
    grad[idx_royalty_ψ(model,obs)] -= sumprod3(ψ1,    qm, cm)        # tmapreduce((q,c,u,v) -> q*c*ψ1(u,v),   +, qm,cm,um,vm; init=zero(eltype(qm)))
    grad[idx_royalty_β(model,obs)] -= dot(qm, cm) .* x
    l > 1 && ( grad[idx_royalty_κ(model,obs,l-1)] -= dot(qm, am) )
    l < L && ( grad[idx_royalty_κ(model,obs,l)]   += dot(qm, bm) )
end

# ---------------------------------------------
# mini version of royalty computations for sake of testing
# ---------------------------------------------


"for testing only"
function llthreads!(grad, θ, RM::RoyaltyModelNoHet, data::DataRoyalty, dograd::Bool=true)

    theta_royalty_check(RM, data, θ) || return -Inf

    k = _num_x(data)
    n = length(data)
    L = _num_choices(data)
    nparm = length(RM,data)

    @assert nparm == length(grad)

    gradtmp = zeros(Float64, nparm, n)
    LL      = Vector{Float64}(undef, n)

    update_xbeta!(data, theta_royalty_β(RM, data, θ))

    let RM=RM, data=data, θ=θ, gradtmp=gradtmp, dograd=dograd, LL=LL
        @threads for i in 1:n
            LL[i] = ll_inner!(view(gradtmp,:,i), data[i], RM, dograd, θ)
        end
    end
    dograd && sum!(reshape(grad, nparm, 1), gradtmp)

    return sum(LL)
end

function ll_inner!(gradtmp::AbstractVector, grp::ObservationGroup{<:DataRoyalty}, RM::AbstractRoyaltyModel, dograd::Bool, θ::AbstractVector)
    obs = first(grp)
    eta12 = η12(obs, RM, θ, _xbeta(obs))
    F, LL = lik_loglik_royalty(obs, eta12)
    l = _y(obs)
    L = _num_choices(obs)

    if dograd
        a, b = normpdf.(eta12) ./ F
        c = dlogcdf_trunc(eta12...)

        gradtmp[idx_royalty_β(RM,obs)] .= - c .* _x(obs)
        l > 1  && ( gradtmp[idx_royalty_κ(RM, obs, l-1)] = -a )
        l < L  && ( gradtmp[idx_royalty_κ(RM, obs, l)  ] =  b )
    end
    return LL
end
