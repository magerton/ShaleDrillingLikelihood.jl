# ---------------------------------------------
# Royalty low level computations
# ---------------------------------------------

function η12(obs::ObservationRoyalty, theta::AbstractVector, zm::Real)
    l = _y(obs)
    L = _num_choices(obs)
    η1 = l == 1 ? typemin(zm) : theta_royalty_κ(obs, theta, l-1) - zm
    η2 = l == L ? typemax(zm) : theta_royalty_κ(obs, theta, l)   - zm
    η1 < η2 || throw(error("η1 < η2 is false"))
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
function simloglik_royalty!(obs::ObservationRoyalty, theta::AbstractVector, sim::SimulationDrawsVector, dograd::Bool=true)

    am  = _am(sim)
    bm  = _bm(sim)
    cm  = _cm(sim)
    LLm = _llm(sim)
    M = _num_sim(sim)

    psi = _ψ1(sim)

    l = _y(obs)
    xbeta = _xbeta(obs)

    βψ = theta_royalty_ψ(obs, theta)

    let xbeta=xbeta, βψ=βψ, obs=obs, theta=theta, dograd=dograd
        @inbounds @threads for m in OneTo(M)
            zm  = xbeta + βψ * psi[m]
            eta12 = η12(obs, theta, zm)

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
    end
    return nothing
end

# ---------------------------------------------
# Royalty gradient
# ---------------------------------------------

"gradient for simulated likelihood when integrating"
function grad_simloglik_royalty!(grad::AbstractVector, obs::ObservationRoyalty, theta::AbstractVector, sim::SimulationDrawsVector)

    length(grad) == length(theta) == _nparm(obs) || throw(DimensionMismatch())

    ψ1 = _ψ1(sim)
    dψ1dρ = _dψ1dθρ(sim)
    qm = _qm(sim) # Pr(um,vm | data) already computed!!!
    am = _am(sim) # computed in likelihood
    bm = _bm(sim) # computed in likelihood
    cm = _cm(sim) # computed in likelihood

    model = _model(obs)
    l = _y(obs)         # discrete choice info
    L = _num_choices(obs)  # discrete choice info
    x = _x(obs)

    βψ = theta_royalty_ψ(obs, theta) # parameters

    # gradient
    grad[idx_royalty_ρ(obs)] -= sumprod3(dψ1dρ, qm, cm) * βψ
    grad[idx_royalty_ψ(obs)] -= sumprod3(ψ1,    qm, cm)
    grad[idx_royalty_β(obs)] -= dot(qm, cm) .* x
    l > 1 && ( grad[idx_royalty_κ(obs,l-1)] -= dot(qm, am) )
    l < L && ( grad[idx_royalty_κ(obs,l)]   += dot(qm, bm) )
end

function simloglik!(grad::AbstractVector, grp::ObservationGroup{<:DataRoyalty}, theta, sim, dograd)
    simloglik_royalty!(first(grp), theta, sim, dograd)
end

function grad_simloglik!(grad, grp::ObservationGroup{<:DataRoyalty}, theta, sim)
    grad_simloglik_royalty!(grad, first(grp), theta, sim)
end

# ---------------------------------------------
# mini version of royalty computations for sake of testing
# ---------------------------------------------


"for testing only"
function llthreads!(grad, θ, data::DataRoyalty{<:RoyaltyModelNoHet}, dograd::Bool=true)

    model = _model(data)
    theta_royalty_check(data, θ) || return -Inf

    k = _num_x(data)
    n = length(data)
    L = _num_choices(data)
    ncoef = _nparm(data)

    ncoef == length(grad) || throw(DimensionMismatch())

    gradtmp = zeros(Float64, ncoef, n)
    LL      = Vector{Float64}(undef, n)

    update_xbeta!(data, theta_royalty_β(data, θ))

    let data=data, θ=θ, gradtmp=gradtmp, dograd=dograd, LL=LL
        @threads for i in 1:n
            LL[i] = ll_inner!(view(gradtmp,:,i), data[i], dograd, θ)
        end
    end
    dograd && sum!(reshape(grad, ncoef, 1), gradtmp)

    return sum(LL)
end

function ll_inner!(gradtmp::AbstractVector, grp::ObservationGroup{<:DataRoyalty}, dograd::Bool, θ::AbstractVector)

    RM = _model(_data(grp))
    obs = first(grp)
    eta12 = η12(obs, θ, _xbeta(obs))
    F, LL = lik_loglik_royalty(obs, eta12)
    l = _y(obs)
    L = _num_choices(obs)

    if dograd
        a, b = normpdf.(eta12) ./ F
        c = dlogcdf_trunc(eta12...)

        gradtmp[idx_royalty_β(obs)] .= - c .* _x(obs)
        l > 1  && ( gradtmp[idx_royalty_κ(obs, l-1)] = -a )
        l < L  && ( gradtmp[idx_royalty_κ(obs, l)  ] =  b )
    end
    return LL
end
