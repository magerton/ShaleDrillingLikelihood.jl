using StatsFuns
using Distributions: _F1
using Base.Threads

# ---------------------------------------------
# Royalty low level computations
# ---------------------------------------------

function η12(model::AbstractRoyaltyModel, theta::AbstractVector, l::Integer, zm::Real)
    L = num_choices(model)
    η1 = l == 1 ? typemin(zm) : theta_roy_κ(model, theta, l-1) - zm
    η2 = l == L ? typemax(zm) : theta_roy_κ(model, theta, l)   - zm
    @assert η1 < η2
    return η1, η2
end

function dlogcdf_trunc(a::Real, b::Real)
    # https://github.com/scipy/scipy/blob/a2ffe09aa751749f2372aa13c19c61b2dec5266f/scipy/stats/_continuous_distns.py
    # https://github.com/JuliaStats/Distributions.jl/blob/master/src/truncated/normal.jl
    # https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf
    a == typemin(a) && return   normpdf(b) / normcdf(b)
    b == typemax(b) && return - normpdf(a) / normccdf(a)
    return - _F1(a/sqrt2, b/sqrt2) / sqrthalfπ
end


function lik_royalty(model::AbstractRoyaltyModel, l::Integer, η12::NTuple{2,Real})
    η1, η2 = η12
    if l == 1
        return normcdf(η2)
    elseif l == num_choices(model)
        return normccdf(η1)
    else
        return normcdf(η2) - normcdf(η1)
    end
end


function loglik_royalty(model::AbstractRoyaltyModel, l::Integer, η12::NTuple{2,Real})
    η1, η2 = η12
    if l == 1
        return normlogcdf(η2)
    elseif l == num_choices(model)
        return normlogccdf(η1)
    else
        return log(normcdf(η2) - normcdf(η1))
    end
    return F
end


function lik_loglik_royalty(model::AbstractRoyaltyModel, l::Integer, η12::NTuple{2,Real})
    F = lik_royalty(model, l, η12)
    η1, η2 = η12
    if l == 1
        return  F, normlogcdf(η2)
    elseif l == num_choices(model)
        return  F, normlogccdf(η1)
    else
        return F, log(F)
    end
end

# ---------------------------------------------
# Royalty computations for individual `i`
# ---------------------------------------------

"Tempvars for royalty simulations with `i`"
struct RoyaltyComputation{V0<:AbstractVector, V1<:Vector, V2<:AbstractVector} <: AbstractIntermediateComputations
    choice::Int # data
    x::V0       # data
    am::V1      # tmp
    bm::V1      # tmp
    cm::V1      # tmp
    LLm::V1     # tmp
    qm::V1      # tmp
    u::V2       # simulation
    v::V2       # simulation

    function RoyaltyComputation(
        choice::Int, x::V0, am::V1, bm::V1, cm::V1, LLm::V1, qm::V1, u::V2, v::V2
    ) where {
        V0<:AbstractVector, V1<:AbstractVector, V2<:AbstractVector
    }
        @assert length(am)==length(bm)==length(cm)==length(qm)==length(LLm)==length(u)==length(v)
        return new{V0,V1,V2}(choice, x, am, bm, cm, LLm, qm, u, v)
    end
end

function RoyaltyComputation(l::AbstractVector, X::AbstractMatrix, am, bm, cm, llm, qm, u, v, i::Integer)
    li = l[i]                     # data
    xi = view(X, :, i)            # data
    um, vm = view.( (u,v), :, i ) # simulations
    return RoyaltyComputation(li, xi, am, bm, cm, llm, qm, um, vm)
end

_choice(rc::RoyaltyComputation) = rc.choice
_x(     rc::RoyaltyComputation) = rc.x
_am(    rc::RoyaltyComputation) = rc.am
_bm(    rc::RoyaltyComputation) = rc.bm
_cm(    rc::RoyaltyComputation) = rc.cm
_LLm(   rc::RoyaltyComputation) = rc.LLm
_qm(    rc::RoyaltyComputation) = rc.qm
_u(     rc::RoyaltyComputation) = rc.u
_v(     rc::RoyaltyComputation) = rc.v

length( rc::RoyaltyComputation ) = length(_u(rc))

"this is the main function. for each ψm, it computes LLm(roy|ψm) and ∇LLm(roy|ψm)"
function loglik_royalty!(rc, model, theta, dograd::Bool)

    am = _am(rc)
    bm = _bm(rc)
    cm = _cm(rc)
    LLm = _LLm(rc)
    u = _u(rc)
    v = _v(rc)
    l = _choice(rc)

    @assert choice_in_model(model, l)

    ρ  = theta_roy_ρ(model,theta)
    βψ = theta_roy_ψ(model,theta)
    βx = theta_roy_β(model,theta)

    xbeta = dot( _x(rc), βx )

    @inbounds @threads for m = 1:length(rc)
        psi = _ψ1(u[m], v[m], ρ)
        zm  = xbeta + βψ * psi
        eta12 = η12(model, theta, l, zm)

        if dograd == false
            LLm[m] = loglik_royalty(model, l, eta12)
        else
            η1, η2 = eta12
            F, LLm[m] = lik_loglik_royalty(model, l, eta12)
            am[m] = normpdf(η1) / F
            bm[m] = normpdf(η2) / F
            cm[m] = dlogcdf_trunc(η1, η2)
        end
    end
    return nothing
end

"integrates over royalty"
function update_grad_roy!(grad::AbstractVector, model::RoyaltyModel, theta::AbstractVector, rc::RoyaltyComputation)

    @assert length(grad) == length(theta) == length(model)

    am = _am(rc) # computed in likelihood
    bm = _bm(rc) # computed in likelihood
    cm = _cm(rc) # computed in likelihood
    um = _u(rc)  # halton draws
    vm = _v(rc)  # halton draws
    qm = _qm(rc) # Pr(um,vm | data) already computed!!!

    l = _choice(rc)         # discrete choice info
    L = num_choices(model)  # discrete choice info

    ρ  = theta_roy_ρ(model, theta) # parameters
    βψ = theta_roy_ψ(model, theta) # parameters

    dψdρ(u,v) = dpsidrho(model, theta, u, v)  # for below
    ψ1(u,v) = _ψ1(u,v,ρ)                      # for below

    # gradient
    grad[idx_roy_ρ(model)] -= sumprod(dψdρ, qm, cm, um, vm) * βψ   # tmapreduce((q,c,u,v) -> q*c*dψdρ(u,v), +, qm,cm,um,vm; init=zero(eltype(qm))) * βψ
    grad[idx_roy_ψ(model)] -= sumprod(ψ1,   qm, cm, um, vm)        # tmapreduce((q,c,u,v) -> q*c*ψ1(u,v),   +, qm,cm,um,vm; init=zero(eltype(qm)))
    grad[idx_roy_β(model)] -= dot(qm, cm) .* _x(rc)
    l > 1 && ( grad[idx_roy_κ(model,l-1)] -= dot(qm, am) )
    l < L && ( grad[idx_roy_κ(model,l)]   += dot(qm, bm) )
end

# TODO would be good to evaluate whether I can do better with tmapreduce
function sumprod(f::Function, x::AbstractArray{T}, y::AbstractArray{T}, u::AbstractArray{T}, v::AbstractArray{T}) where {T<:Real}
    @assert length(x)==length(y)==length(u)==length(v)
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i] * f(u[i], v[i])
    end
    return s
end

# ---------------------------------------------
# mini version of royalty computations for sake of testing
# ---------------------------------------------


"for testing only"
function llthreads!(grad, θ, RM::AbstractRoyaltyModel, l, xx, dograd::Bool=true)

    theta_roy_check(RM, θ) || return -Inf

    k,n = size(xx)
    nparm = length(RM)
    @assert nparm == length(grad)

    L = num_choices(RM)

    gradtmp = zeros(Float64, nparm, n)
    LL      = Vector{Float64}(undef, n)

    z = xx' * theta_roy_β(RM, θ)
    # mul!(z, xx', theta_roy_β(RM, θ))

    @threads for i in 1:n
        eta12 = η12(RM, θ, l[i], z[i])
        F, LL[i] = lik_loglik_royalty(RM, l[i], eta12)

        if dograd
            a, b = normpdf.(eta12) ./ F
            c = dlogcdf_trunc(eta12...)

            gradtmp[idx_roy_β(RM), i] .= - c .* view(xx, :, i)
            l[i] > 1  && ( gradtmp[idx_roy_κ(RM, l[i]-1), i] = -a )
            l[i] < L  && ( gradtmp[idx_roy_κ(RM, l[i]),   i] =  b )
        end
    end

    dograd && sum!(reshape(grad, nparm, 1), gradtmp)

    return sum(LL)
end
