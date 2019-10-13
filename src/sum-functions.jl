"""
    logsumexp_and_softmax!(r, x)

Set `r` = softmax(x) and return `logsumexp(x)`.
"""
function logsumexp_and_softmax!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    length(r) == n || throw(DimensionMismatch())
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



# TODO would be good to evaluate whether I can do better with tmapreduce
"""
    sumprod(f, x, y, u, v) = sum(x .* y .* f.(u, v))
"""
function sumprod(f::Function, x::AbstractArray{T}, y::AbstractArray{T}, u::AbstractArray{T}, v::AbstractArray{T}) where {T<:Real}
    length(x)==length(y)==length(u)==length(v) || throw(DimensionMismatch())
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i] * f(u[i], v[i])
    end
    return s
end


# https://github.com/jw3126/ThreadingTools.jl
# https://discourse.julialang.org/t/how-to-speed-up-this-simple-code-multithreading-simd-inbounds/19681/39
# https://discourse.julialang.org/t/parallelizing-for-loop-in-the-computation-of-a-gradient/9154/7?u=tkoolen
# https://discourse.julialang.org/t/innefficient-paralellization-need-some-help-optimizing-a-simple-dot-product/9723/31
# https://discourse.julialang.org/t/for-loops-acceleration/24011/7

# tmapreduce((q,c,u,v) -> q*c*dψdρ(u,v), +, qm,cm,um,vm; init=zero(eltype(qm))) * βψ

"""
    sumprod3(x, y, z)

sumprod3 = sum( x .* y .* z)
"""
function sumprod3(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T<:Real}
    length(x)==length(y)==length(z) || throw(DimensionMismatch())
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i] * z[i]
    end
    return s
end
