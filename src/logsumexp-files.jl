"""
    logsumexp_and_softmax!(r, x)

Set `r` = softmax(x) and return `logsumexp(x)`.
"""
function logsumexp_and_softmax!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    @assert length(r) == n
    isempty(x) && return -T(Inf)

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

    s = zero(T)
    @inbounds @simd for i in 1:n
        s += ( r[i] = exp(x[i] - u) )
    end
    invs = one(T)/s
    r .*= invs
    return log(s) + u
end

logsumexp_and_softmax!(x) = logsumexp_and_softmax!(x,x)

function flow(d::Integer, theta::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(T)
    return T(theta[d-1]*x + theta[d+1]*ψ)
end

function dflow(k::Integer, d::Integer, thet::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    @assert 1 <= k <= length(thet)
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
