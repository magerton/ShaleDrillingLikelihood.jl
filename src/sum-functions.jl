struct WeightedParticles{T,N}
    particles::Vector{T}
    # weights::Vector{T}
    logweights::Vector{T}
end
# See https://discourse.julialang.org/t/fast-logsumexp/22827/6
# See https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/4f9b688d298157dc24a5b0a518d971221fbe15dd/src/resampling.jl#L1-L24

function logsumexp!(w,we)
    offset = maximum(w)
    we .= exp.(w .- offset)
    s = sum(we)
    w .-= log(s) + offset
    we .*= 1/s
end

@generated function logsumexp_loopvectorization!(w::Vector{T},we) where T
    quote
        offset = maximum(w)
        N = length(w)
        s = zero(T)
        @vectorize $T for i = 1:N
            wl = w[i]
            wel = exp(wl-offset)
            we[i] = wel
            s += wel
        end
        w  .-= log(s) + offset
        we .*= 1/s
    end
end

"""
    logΣexp, Σ = logsumexp!(p::WeightedParticles)
Return log(∑exp(w)). Modifies the weight vector to `w = exp(w-offset)`
Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow.
References:
https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(p::WeightedParticles)
    N = length(p)
    w = p.logweights
    offset, maxind = findmax(w)
    w .= exp.(w .- offset)
    Σ = sum_all_but(w,maxind) # Σ = ∑wₑ-1
    log1p(Σ) + offset, Σ+1
end

function sum_all_but(w,i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    s
end


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
