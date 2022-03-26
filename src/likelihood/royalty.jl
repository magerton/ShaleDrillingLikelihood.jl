# ---------------------------------------------
# Royalty low level computations
# ---------------------------------------------

function η12(obs::ObservationRoyalty, theta::AbstractVector, zm::Real)
    l = _y(obs)
    # L = num_choices(obs)
    # η1 = l == 1 ? typemin(zm) : theta_royalty_κ(obs, theta, l-1) - zm
    # η2 = l == L ? typemax(zm) : theta_royalty_κ(obs, theta, l)   - zm
    η1, η2 = kappa_sums(obs, theta) .- zm
    η1 < η2 || throw(error("$η1 = η1 < η2=$η2 is false"))
    return η1, η2
end

@inline function kappa_sums(obs, theta::AbstractVector{T})::NTuple{2,T} where {T}
    l = _y(obs)
    L = num_choices(obs)

    kappas = theta_royalty_κ(obs, theta)
    k0 = first(kappas)
    ksum = 0.5*sumsq(kappas, 2, l-1)

    if l == 1
        return (typemin(T), k0)
    elseif 1 < l < L
        tmp = k0+ksum
        klsq = 0.5*kappas[l]^2
        return (tmp, tmp+klsq)
    elseif l == L
        return (k0+ksum, typemax(T))
    else
        throw(DomainError(l))
    end
end

# see
# https://github.com/cossio/TruncatedNormal.jl/blob/fc904152f2da11a257e3ccdd3e49ef118b81d437/notes/normal.pdf
# https://stats.stackexchange.com/questions/7200/evaluate-definite-interval-of-normal-distribution/7206#7206

function dlogcdf_trunc(a::Real, b::Real)
    # https://github.com/scipy/scipy/blob/a2ffe09aa751749f2372aa13c19c61b2dec5266f/scipy/stats/_continuous_distns.py
    # https://github.com/JuliaStats/Distributions.jl/blob/master/src/truncated/normal.jl
    # https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf
    a == typemin(a) && return   normpdf(b) / normcdf(b)
    b == typemax(b) && return - normpdf(a) / normccdf(a)
    return -_tnmom1(a,b)
end


function lik_royalty(obs::ObservationRoyalty, η12::NTuple{2,Real})
    η1, η2 = η12
    l = _y(obs)
    if l == 1
        return normcdf(η2)
    elseif l == num_choices(obs)
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
    elseif l == num_choices(obs)
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
    elseif l == num_choices(obs)
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
    isfinite(xbeta) || throw(error("xbeta = $xbeta not finite"))

    check_finite(theta)

    βψ = theta_royalty_ψ(obs, theta)
    @inbounds for m in OneTo(M)
        zm  = xbeta + βψ * psi[m]
        eta12 = η12(obs, theta, zm)

        if dograd == false
            LL = loglik_royalty(obs, eta12)
            isfinite(LL) || @warn "LL = $LL not finite"
            LLm[m] += LL
        else
            η1, η2 = eta12
            F,LL  = lik_loglik_royalty(obs, eta12)
            isfinite(LL) || @warn "LL = $LL not finite. η1, η2 = $eta12 and choice =$l"
            isfinite(F) || F <= 0 || @warn "F = $F not finite or zero"
            LLm[m] += LL

            
            # η1, η2 .=  (κlo,κhi) .- zm
            # bqm[m] = normpdf(η2) / normcdf(η2)
            # aqm[m] = normpdf(η1) / (1-normcdf(η1))
            # cqm[m]  = - (normpdf(η1) - normpdf(η2)) / (normpdf(η1) - normpef(η2))
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
    L = num_choices(obs)  # discrete choice info
    x = _x(obs)

    βψ = theta_royalty_ψ(obs, theta) # parameters

    aqm = dot(qm,am)
    bqm = dot(qm,bm)
    cqm = dot(qm,cm)

    # gradient
    grad[idx_royalty_ρ(obs)] -= sumprod3(dψ1dρ, qm, cm) * βψ
    grad[idx_royalty_ψ(obs)] -= sumprod3(ψ1,    qm, cm)
    grad[idx_royalty_β(obs)] -= cqm .* x

    if l == 1
        grad[idx_royalty_κ(obs,l)] += bqm
    elseif l < L
        idx = 2:l-1
        grad[idx_royalty_κ(obs,1)]    += cqm
        grad[idx_royalty_κ(obs,idx)] .+= cqm .* theta_royalty_κ(obs, theta, idx)
        grad[idx_royalty_κ(obs,l)]    += bqm  * theta_royalty_κ(obs, theta, l)
    elseif l == L
        idx = 2:l-1
        grad[idx_royalty_κ(obs,1)]   -= aqm
        grad[idx_royalty_κ(obs,idx)] -= aqm .* theta_royalty_κ(obs, theta, idx)
    else
        throw(DomainError(l))
    end
end

function simloglik!(grad::AbstractVector, grp::ObservationGroupRoyalty, theta, sim, dograd; kwargs...)
    simloglik_royalty!(first(grp), theta, sim, dograd)
end

function grad_simloglik!(grad, grp::ObservationGroupRoyalty, theta, sim)
    grad_simloglik_royalty!(grad, first(grp), theta, sim)
end

# ---------------------------------------------
# mini version of royalty computations for sake of testing
# ---------------------------------------------


"for testing only"
function llthreads!(grad, θ, data::DataRoyalty{<:RoyaltyModelNoHet}, dograd::Bool=true)

    model = _model(data)
    # theta_royalty_check(data, θ) || return -Inf

    k = _num_x(data)
    n = length(data)
    L = num_choices(data)
    ncoef = _nparm(data)

    ncoef == length(grad) || throw(DimensionMismatch())

    gradtmp = zeros(Float64, ncoef, n)
    LL      = Vector{Float64}(undef, n)

    beta = theta_royalty_β(data, θ)
    check_finite(beta)

    update_xbeta!(data, beta)

    for i in OneTo(n)
        gtmp = uview(gradtmp, :, i)
        LL[i] = ll_inner!(gtmp, data[i], dograd, θ)
    end
    dograd && sum!(reshape(grad, ncoef, 1), gradtmp)

    return sum(LL)
end

function ll_inner!(gradtmp::AbstractVector, grp::ObservationGroupRoyalty, dograd::Bool, θ::AbstractVector)

    RM = _model(_data(grp))
    obs = first(grp)
    eta12 = η12(obs, θ, _xbeta(obs))
    F, LL = lik_loglik_royalty(obs, eta12)
    l = _y(obs)
    L = num_choices(obs)
    theta = θ

    if dograd
        a, b = normpdf.(eta12) ./ F
        c = dlogcdf_trunc(eta12...)

        gradtmp[idx_royalty_β(obs)] .= - c .* _x(obs)
        # l > 1  && ( gradtmp[idx_royalty_κ(obs, l-1)] = -a )
        # l < L  && ( gradtmp[idx_royalty_κ(obs, l)  ] =  b )

        if l == 1
            gradtmp[idx_royalty_κ(obs,l)] += b
        elseif l < L
            idx = 2:l-1
            gradtmp[idx_royalty_κ(obs,1)]    += c
            gradtmp[idx_royalty_κ(obs,idx)] .+= c .* theta_royalty_κ(obs, theta, idx)
            gradtmp[idx_royalty_κ(obs,l)]    += b  * theta_royalty_κ(obs, theta, l)
        elseif l == L
            idx = 2:l-1
            gradtmp[idx_royalty_κ(obs,1)]   -= a
            gradtmp[idx_royalty_κ(obs,idx)] -= a .* theta_royalty_κ(obs, theta, idx)
        else
            throw(DomainError(l))
        end
    end
    return LL
end

function solve_model(data::DataRoyalty{RoyaltyModelNoHet}, theta)

    grad = similar(theta)
    f(parm) = - llthreads!(grad, parm, data, false)
    function fg!(g, parm)
        LL = llthreads!(g, parm, data, true)
        g .*= -1
        return -LL
    end

    odfg = OnceDifferentiable(f, fg!, fg!, theta)

    # check that it solves
    opts = Optim.Options(time_limit = 60, allow_f_increases=true)
    res = optimize(odfg, theta, BFGS(), opts)
    return res
end
