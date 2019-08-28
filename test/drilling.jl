# module ShaleDrillingLikelihood_Drilling_Test
#
# using ShaleDrillingLikelihood
# using Test
using StatsFuns
using Distributions
using Random
using BenchmarkTools
using Base.Threads

# using Calculus
# using Optim
# using LinearAlgebra

# @testset "Drilling Likelihood" begin


# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

# -----------------------------------------------
# simulate data
# -----------------------------------------------

Random.seed!(1234)

num_i = 10_000
num_t = 500
nobs = num_i *num_t
β = [1.0, -2.0, 1.0, 2.0]
k = length(β)
L = 3

# exogenous variable
X = randn(nobs)
psi = repeat(randn(num_i), inner=num_t)

# choice utilities
u = vcat(zeros(nobs)', β[1:2]*X' .+ β[3:4]*psi')

# multinomial logit probabilities
Pr0 = mapslices(softmax, u; dims=1)
cum_Pr0 = cumsum(Pr0; dims=1)

# random choice given multinomial probabilites
e_quantile = rand(nobs)
choices = [searchsortedfirst(view(cum_Pr0, :, i), e_quantile[i]) for i in 1:length(e_quantile) ]

# -----------------------------------------------
# form likelihood
# -----------------------------------------------

theta = rand(k)
ubV = zeros(L, nobs)
drng = 1:L
M = 100
psisim = randn(M,num_i)

function flow(d::Integer,theta::AbstractVector,x::Real,ψ::Real,L::Integer)
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(x)
    return theta[d-1]*x + theta[d+1]*ψ
end

function loglik_i(yi::AbstractVector{Int}, xi::AbstractVector{<:Real}, psii::AbstractVector{<:Real}, theta::AbstractVector{<:Real}, ubv::AbstractArray{<:Real}, llm::AbstractVector{T}, L::Integer) where {T}
    M = length(psii)
    L = size(ubv,1)
    num_t = length(yi)
    @assert num_t == length(xi)
    @assert M == length(llm)

    fill!(llm, zero(T))

    for t in 1:num_t
        for m in 1:M
            @simd for d in 1:L
                @inbounds ubv[d] = flow(d, theta, xi[t], psii[m], L)
            end
            d_choice = yi[t]
            @views llm[m] += ubv[d_choice] - logsumexp(ubv)
        end
    end
    return logsumexp(llm)
end


irng(num_t::Int,i::Int) = ((i-1)*num_t) .+ 1:num_t

function loglik(y::AbstractVector, x::AbstractArray, psi::AbstractArray, theta::AbstractVector{T}, ubv::AbstractArray, llm::AbstractArray, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    @assert size(ubv) == (L,nthreads())
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i

    LL = Atomic{T}(zero(T))
    for i in 1:num_i
        rng = irng(num_t,i)
        @views atomic_add!(LL, loglik_i(y[rng], x[rng], psi[:,i], theta, ubv[:,threadid()], llm[:,threadid()], L))
    end
    return LL[]
end

ubvtmp = Array{Float64}(undef, L, nthreads())
llmtmp = Array{Float64}(undef, M, nthreads())

f(θ) = -loglik(choices, X, psisim, θ, ubvtmp, llmtmp, num_t, num_i)
@btime f(theta)



@code_warntype loglik(choices, X, psisim, theta, ubvtmp, llmtmp, num_t, num_i)
let y = choices[irng(num_t,1)], x = X[irng(num_t,1)], psii = psisim[:,1], ubv = ubvtmp[:,1], llm = llmtmp[:,1]
    @code_warntype loglik_i(y, x, psii, theta, ubv, llm, L)
end


# res = optimize(f, theta, BFGS(), Optim.Options(time_limit=60.0*5, show_trace=true))
#
# @show res
# @show res.minimizer

# end
#
# end # module
