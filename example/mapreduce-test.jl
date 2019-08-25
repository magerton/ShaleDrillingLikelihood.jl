# dev ssh://git@github.com/mohamed82008/KissThreading.jl.git
# add Strided

using Base.Threads
using KissThreading
using Strided
using BenchmarkTools
using Test

n = 1_000
X = rand(n,n) .* 0.1
Y = rand(n,n) .* 0.1
Z = rand(n,n) .* 0.1

# function sumprod_manual(X::Array{T},Y::Array{T}) where {T}
#     size(X)==size(Y) || throw(DimensionMismatch())
#     s = zero(T)
#     @inbounds @simd for i in eachindex(X)
#         s += X[i]*Y[i]
#     end
#     return s
# end

function sumthree(f::Function, x::AbstractArray{T}, y::AbstractArray{T}, u::AbstractArray{T}) where {T<:Real}
    @assert size(x)==size(y)==size(u)
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i] * u[i]
    end
    return s
end

function sumprod_inbounds_simd_fastmath(X::Array{T}, Y::Array{T}, Z::Array{T}) where {T}
    @assert size(X)==size(Y)==size(Z)
    s = zero(T)
    @inbounds @simd for i in eachindex(X)
        s += @fastmath X[i]*Y[i]*Z[i]
    end
    return s
end

function sumprod_inbounds_simd(X::Array{T}, Y::Array{T}, Z::Array{T}) where {T}
    @assert size(X)==size(Y)==size(Z)
    s = zero(T)
    @inbounds @simd for i in eachindex(X)
        s += X[i]*Y[i]*Z[i]
    end
    return s
end


function sumprod_inbounds(X::Array{T}, Y::Array{T}, Z::Array{T}) where {T}
    @assert size(X)==size(Y)==size(Z)
    s = zero(T)
    @inbounds for i in eachindex(X)
        s += X[i]*Y[i]*Z[i]
    end
    return s
end

function sumprod(X::Array{T}, Y::Array{T}, Z::Array{T}) where {T}
    @assert size(X)==size(Y)==size(Z)
    s = zero(T)
    for i in eachindex(X)
        s += X[i]*Y[i]*Z[i]
    end
    return s
end


println("\n\nTHREE matrices")

println("mapreduce")
@btime mapreduce(prod, +, zip(X,Y,Z))

println("kiss tmapreduce")
@btime tmapreduce((x,y,z) -> x*y*z, +, X,Y,Z; init=zero(eltype(X)))

println("loop: inbounds, simd, fastmath")
@btime sumprod_inbounds_simd_fastmath(X,Y,Z)

println("loop: inbounds, simd")
@btime sumprod_inbounds_simd(X,Y,Z)

println("loop: inbounds")
@btime sumprod_inbounds(X,Y,Z)

println("loop")
@btime sumprod(X,Y,Z)

println("sumthree")
@btime sumthree((xi,yi,zi) -> xi*yi*zi, X,Y,Z)

println("broadcast")
@btime sum(X.*Y.*Z)

println("strided mapreduce")
@btime @strided mapreduce(prod, +, zip(X, Y, Z))


# time rankings
println("\n\nONE matrix")
@btime  mapreduce((x) -> x^2, +, StridedView(X))
@btime  mapreduce((x) -> x^2, +, X)
@btime tmapreduce((x) -> x^2, +, X,; init=zero(eltype(X)))
@btime sum(X.^2)

# 2 matrices
println("\n\nTWO matrices")
@btime @strided mapreduce(prod, +, zip(X, Y))
@btime          mapreduce(prod, +, zip(X,Y))
@btime tmapreduce((x,y) -> x*y, +, X, Y; init=zero(eltype(X)))
@btime sum(X.*Y)


# println("strided")
# @btime @strided mapreduce((x) -> x^2, +, X)
# println("kiss")
# @btime tmapreduce((x) -> x^2, +, X; init=zero(eltype(X)))
# println("regular")
# @btime mapreduce((x) -> x^2, +, X)
