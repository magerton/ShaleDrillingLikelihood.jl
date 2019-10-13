function flow(d::Integer, theta::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    1 <= d <= L  || throw(BoundsError()) # FIXME with L
    d==1 && return zero(T)
    return T(theta[d-1]*x + theta[d+1]*ψ)
end

function dflow(k::Integer, d::Integer, thet::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    1 <= d <= L || throw(BoundsError())
    1 <= k <= length(thet) || throw(BoundsError())
    d==1 && return zero(T)
    if d == 2
        k == 1 && return T(x)
        k == 3 && return T(ψ)
        return zero(T)
    else
        k == 2 && return T(x)
        k == 4 && return T(ψ)
        return zero(T)
    end
end
