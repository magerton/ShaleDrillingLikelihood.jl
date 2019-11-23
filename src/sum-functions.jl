export logsumexp!, softmax3!, logsumexp_and_softmax!



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



# TODO would be good to evaluate whether I can do better with tmapreduce
# https://github.com/jw3126/ThreadingTools.jl
# https://discourse.julialang.org/t/how-to-speed-up-this-simple-code-multithreading-simd-inbounds/19681/39
# https://discourse.julialang.org/t/parallelizing-for-loop-in-the-computation-of-a-gradient/9154/7?u=tkoolen
# https://discourse.julialang.org/t/innefficient-paralellization-need-some-help-optimizing-a-simple-dot-product/9723/31
# https://discourse.julialang.org/t/for-loops-acceleration/24011/7

# tmapreduce((q,c,u,v) -> q*c*dψdρ(u,v), +, qm,cm,um,vm; init=zero(eltype(qm))) * βψ

# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39


"""
    logsumexp!(r, x)

Compute `r` = softmax(x) and return `logsumexp(x)`.

Based on code from
https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
@generated function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    quote
        n = length(x)
        length(r) == n || throw(DimensionMismatch())
        isempty(x) && return -T(Inf)
        1 == stride1(r) == stride1(x) || throw(error("Arrays not strided"))

        u = maximum(x)                                       # max value used to re-center
        abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

        s = zero(T)
        @vectorize $T for i = 1:n
            tmp = exp(x[i] - u)
            r[i] = tmp
            s += tmp
        end

        invs = inv(s)
        r .*= invs

        return log1p(s-1) + u
    end
end

logsumexp!(x) = logsumexp!(x,x)

# @deprecate logsumexp_and_softmax!(x) logsumexp!(x)

add_1_dim(x::AbstractArray) = reshape(x, size(x)..., 1)

"""
    softmax3!(q, lse, tmp, x, maxk)

set `q[i,j,k] = Pr( k | x[i,j,:])` for k in 1:maxk

Sets `lse = ∑_k exp(x[i,j,k])`

Uses temporary array `tmp`
"""
@generated function softmax3!(q::AA, lse::Matrix{T}, tmpmax::Matrix{T}, x::AA, maxk::Integer=size(q, ndims(q)) ) where {T<:Real, AA<:AbstractArray{T}}
    quote
        xsizes = size(x)
        xsizes == size(q) || throw(DimensionMismatch("size(x) = $(size(x)) but size(q) = $(size(q))"))
        nk = last(xsizes)
        xsizes[1:end-1] ==  size(lse) == size(tmpmax) || throw(DimensionMismatch("size(x) = $(size(x)),  size(lse) = $(size(lse)), and size(tmpmax) = $(size(tmpmax))"))
        0 < maxk <= nk || throw(DomainError(maxk))
        1 == stride1(q) == stride1(x) || throw(error("Arrays not strided"))

        isempty(x) && throw(error("x empty"))
        all(isfinite.(x)) || throw(error("x not finite"))

        maximum!(add_1_dim(tmpmax), x)
        fill!(lse, zero(T))

        xx = reshape(x, :, nk)
        qq = reshape(q, :, nk)

        for k in OneTo(nk)
            @vectorize $T for i = 1:length(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                k <= maxk && (qq[i,k] = tmp)
            end
        end
        qq[:,OneTo(maxk)] ./= vec(lse)
    end
end

"use in vfit!"
function logsumexp_and_softmax!(lse, q, tmpmax, x, args...)
    softmax3!(q, lse, tmpmax, x, args...)
    lse .-= 1
    lse .= log1p.(lse) .+ tmpmax
end


function softmax3test!(q::AbstractArray3, x::AbstractArray3)
    ni, nj, nk = size(x)
    (ni, nj, nk) == size(q) || throw(DimensionMismatch())
    @inbounds for j = OneTo(nj)
        for i in OneTo(ni)
            @views softmax!(q[i,j,:], x[i,j,:])
        end
    end
end

function logsumexp3test!(x::AbstractArray3{T}) where {T}
    ni, nj, nk = size(x)
    lse = zeros(T,ni,nj)
    tmp = zeros(T,nk)
    # @inbounds
    for j = OneTo(nj)
        for i in OneTo(ni)
            tmp .= view(x, i, j, :)
            lse[i,j] = logsumexp!(tmp)
        end
    end
    return lse
end
