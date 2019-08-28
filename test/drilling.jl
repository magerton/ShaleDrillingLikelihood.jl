# module ShaleDrillingLikelihood_Drilling_Test
#
# using ShaleDrillingLikelihood
# using Test
using StatsFuns
using Distributions
# using Calculus
using Optim
using Random
using LinearAlgebra
#
# @testset "Drilling Likelihood" begin


# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

# -----------------------------------------------
# simulate data
# -----------------------------------------------

Random.seed!(1234)

num_i = 1_000
num_t = 25
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

function loglik_i(yi::AbstractVector{Int}, xi::AbstractVector{<:Real}, psii::AbstractVector{<:Real}, theta::AbstractVector{<:Real}, ubv::AbstractVector{<:Real}, llm::AbstractVector{T}, L::Integer) where {T}
    M = length(psii)
    L = length(ubv)
    num_t = length(yi)
    @assert num_t == length(xi)
    @assert M == length(llm)

    fill!(llm, zero(T))

    for t in 1:num_t
        for m in 1:M
            @inbounds @simd for d in 1:L
                ubv[d] = flow(d, theta, xi[t], psii[m], L)
            end
            d_choice = yi[t]
            llm[m] += ubv[d_choice] - logsumexp(ubv)
        end
    end
    return logsumexp(llm)
end


function loglik(y, x, psi, theta::AbstractVector{T}, ubv, llm, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    @assert length(ubv) == L
    @assert minimum(y) == 1
    M = size(psi,1)
    @assert size(psi,2) == num_i

    LL = zero(T)

    for i in 1:num_i
        irng = ((i-1)*num_t) .+ 1:num_t
        @views LL += loglik_i(y[irng], x[irng], psi[:,i], theta, ubv, llm, L)
    end
    return LL
end

ubvtmp = Vector{Float64}(undef, L)
llmtmp = Vector{Float64}(undef, M)

f(θ) = -loglik(choices, X, psisim, θ, ubvtmp, llmtmp, num_t, num_i)

res = optimize(f, theta, BFGS(), Optim.Options(time_limit=60.0*5, show_trace=true))

@show res
@show res.minimizer

# end
#
# end # module
