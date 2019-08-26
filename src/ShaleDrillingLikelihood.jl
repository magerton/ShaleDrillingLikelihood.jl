module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra

abstract type AbstractIntermediateComputations end


# TODO would be good to evaluate whether I can do better with tmapreduce
function sumprod(f::Function, x::AbstractArray{T}, y::AbstractArray{T}, u::AbstractArray{T}, v::AbstractArray{T}) where {T<:Real}
    @assert length(x)==length(y)==length(u)==length(v)
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

function sumprod3(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T<:Real}
    @assert length(x)==length(y)==length(z)
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i] * z[i]
    end
    return s
end




include("tmp.jl")
include("models.jl")
include("royalty.jl")
include("production.jl")
# include("drilling.jl")

end # module
