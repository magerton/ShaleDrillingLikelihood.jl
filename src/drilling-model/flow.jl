Dgt0(m::AbstractDrillModel, state::Integer) = throw(error("Dgt0 not defined for $(m)"))
Dgt0(m::TestDrillModel,     state::Integer) = state > 1

# ------------------------------------------

@inline _ρ(θρ) = θρ # logistic(θρ)

@inline _dρdθρ(θρ::T) = one(T) # (z = logistic(θρ); z*(1-z) )
@deprecate _dρdσ(θρ) _dρdθρ(θρ)

@inline _ρsq(θρ) = _ρ(θρ)^2
@deprecate _ρ2(θρ) _ρsq(θρ)

@inline _ψ1(u::Real,v::Real,ρ::Real) = ρ*u + sqrt(1-ρ^2)*v
@inline _ψ2(u::Real,v::Real, ρ::Real) = _ψ2(u,v)
@inline _ψ2(u::Real,v::Real) = u

@inline _dψ1dρ(u::Real,v::Real,ρ::Real) = u - ρ/sqrt(1-ρ^2)*v

@inline _dψ1dθρ(u::Real, v::Real, ρ::Real, θρ::Real) = _dψ1dρ(u,v,ρ)*_dρdθρ(θρ::Real)

@inline _z(x2::Real, x1::Real, Δ::Real, ρ::Real) = (x2 - ρ*x1 + Δ)/sqrt(1-ρ^2)

# derivatives
@inline _dzdρ(x2::Real, x1::Real, ρ::Real, z::Real) = -x1/sqrt(1-ρ^2) + ρ*z/(1-ρ^2)
@inline _dπdρ(x2::Real, x1::Real, Δ::Real, ρ::Real) = (z = _z(x2,x1,Δ,ρ);  normpdf(z) * _dzdρ(x2,x1,ρ,z))

# finite difference versions
@inline _ρ(x::Real, h::Real) = _ρ(x+h)
@inline _z(x2::Real, x1::Real, Δ::Real, ρ::Real, h::Real) = _z(x2, x1+h, Δ, ρ)

# -------------------------------------------------------

function check_model_dims(d::Integer, obs::ObservationDrill, theta::AbstractVector)
    model = _model(obs)
    length(model) == length(theta) || throw(DimensionMismatch())
    d in actionspace(model, _x(obs)) || throw(BoundsError())
end




function flow(d::Integer, obs::ObservationDrill{TestDrillModel}, theta::AbstractVector{T}, psis::NTuple{2}) where {T}
    check_model_dims(d,obs,theta)
    if d == 0
        payoff = 0.0
    else
        x = _x(obs)
        z = zchars(obs)
        psi = Dgt0(model, x) : last(psis) : first(psis)
        payoff = d*(theta[1]*psi + theta[2]*x + theta[3]*first(z))
    end
    return T(payoff)
end



function dflow(k::Integer, d::Integer, obs::ObservationDrill{TestDrillModel}, theta::AbstractVector{T}, psis::NTuple{2}) where {T}
    check_model_dims(d,obs,theta)
    1 <= k <= length(theta) || throw(BoundsError(theta,k))

    if d == 0
        return zero(T)
    else
        x = _x(obs)
        z = zchars(obs)
        psi = Dgt0(model, x) : last(psis) : first(psis)
        k == 1 && return T(d*psi)
        k == 2 && return T(d*x)
        k == 3 && return T(d*first(z))
    end
end

function dflowdψ(d::Integer, obs::ObservationDrill{TestDrillModel}, theta::AbstractVector{T}, psis::NTuple{2}) where {T}
    check_model_dims(d,obs,theta)
    if d == 0
        return zero(T)
    else
        return T(d*theta[1])
    end
end

function dflowdrho(d::Integer, obs::ObservationDrill{TestDrillModel}, theta::AbstractVector{T}, psis::NTuple{2}) where {T}
    check_model_dims(d,obs,theta)
    if d == 0
        return zero(T)
    else
        return T(d*theta[1])
    end
end
