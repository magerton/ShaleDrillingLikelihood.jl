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

    η1 < η2 || throw(error("η1 < η2 is FALSE. $η1 ≰ $η2"))
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
struct RoyaltyComputation{V0<:AbstractVector, V1<:AbstractVector, V2<:AbstractVector} <: AbstractIntermediateComputations
    choice::Int

    x::V0

    am::V1
    bm::V1
    cm::V1
    LLm::V1
    qm::V1

    u::V2
    v::V2

    # inner constructor checks the length
    function RoyaltyComputation(
        choice::Int,
        x::V0,
        am::V1, bm::V1, cm::V1, LLm::V1, qm::V1,
        u::V2, v::V2
    ) where {
        V0<:AbstractVector, V1<:AbstractVector, V2<:AbstractVector
    }

        length(am)==length(bm)==length(cm)==
            length(qm) == length(LLm)==
            length(u)==length(v) || throw(DimensionMismatch())

        return new{V0,V1,V2}(choice, x, am, bm, cm, LLm, qm, u, v)
    end
end


# TODO make tempvars
function RoyaltyComputation(l::AbstractVector, X::AbstractMatrix, a::Matrix, b::Matrix, c::Matrix, LL::Matrix, Q::Matrix, u::Matrix, v::Matrix, i::Integer)

    # data
    li = l[i]
    xi = view(X, :, i)

    # tmpvars
    am, bm, cm, LL, qm, = view.( (a,b,c,LL,Q), :, i )

    # simulations
    u, v = view.( (u,v), :, i )
    return RoyaltyComputation(li, xi, am, bm, cm, LL, qm, u, v)
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

length( rc::RoyaltyComputation) = length(_u(rc))


"this is the main function. for each ψm, it computes LLm(roy|ψm) and ∇LLm(roy|ψm)"
function loglik_royalty!(rc, model, theta, dograd::Bool)

    am = _am(rc)
    bm = _bm(rc)
    cm = _cm(rc)
    LLm = _LLm(rc)

    u = _u(rc)
    v = _v(rc)

    l = _choice(rc)

    choice_in_model(model, l) || throw(DomainError(l))

    ρ  = theta_roy_ρ(model,theta)
    βψ = theta_roy_ψ(model, theta)
    βx = theta_roy_β(model,theta)

    xbeta = dot( _x(rc), βx )

    for m = 1:length(rc)
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
            cm[m] = dlogcdf_trunc(η1,η2)
        end
    end

    return nothing
end

"integrates over royalty"
function update_grad_royalty!(grad::AbstractVector, model::AbstractRoyaltyModel, theta::AbstractVector, rc::RoyaltyComputation)

    # unpack stuff
    am = _am(rc)
    bm = _bm(rc)
    cm = _cm(rc)
    um = _u(rc)
    vm = _v(rc)
    qm = _qm(rc)

    # parameters
    ρ  = theta_roy_ρ(model,theta)
    βψ = theta_roy_ψ(model, theta)

    l = _choice(rc)
    L = num_choices(model)

    # gradient
    grad[idx_roy_ρ(model)] -= sum(qm .* cm .* dpsidrho.((model,), (theta,), um, vm) ) * βψ
    grad[idx_roy_ψ(model)] -= sum(qm .* cm .* _ψ1.(um, vm, ρ))
    grad[idx_roy_β(model)] -= dot(qm, cm) * _x(rc)

    l > 1 && ( grad[idx_roy_κ(model,l-1)] -= sum(qm .* am) )
    l < L && ( grad[idx_roy_κ(model,l)]   += sum(qm .* bm) )
end

# ---------------------------------------------
# mini version of royalty computations for sake of testing
# ---------------------------------------------


"for testing only"
function llthreads!(grad, θ, RM::AbstractRoyaltyModel, l, xx, dograd::Bool=true)

    theta_roy_check(RM, θ) || return -Inf

    k,n = size(xx)
    nparm = length(RM)
    nparm == length(grad) || throw(DimensionMismatch())

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
