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

function flow(d::Integer,theta::AbstractVector,x::Real,ψ::Real)
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(x)
    return theta[d-1]*x + theta[d+1]*ψ
end


function loglik(y, x, psi, theta, ubv, llm, num_t)
    M, num_i = size(psi)
    @assert (1,3,) == extrema(y)
    @assert length(ubv) == 3
    length(y) == length(x) == num_i * num_t

    LL = 0.0
    for i in 1:num_i
        fill!(llm, 0.0)
        for m in 1:M
            for t in 1:num_t
                n = (i-1)*num_t + t
                @inbounds @simd for d in 1:3
                    ubv[d] = flow(d,theta,x[n],psi[m,i])
                end
                llm[m] += ubv[y[n]] - logsumexp(ubv)
            end
            LL += logsumexp(llm)
        end
    end
    return LL
end

ubvtmp = Vector{Float64}(undef, L)
llmtmp = Vector{Float64}(undef, M)

f(θ) = -loglik(choices, X, psisim, θ, ubvtmp, llmtmp, num_t)

res = optimize(f, theta, NelderMead(), Optim.Options(time_limit=100.0, show_trace=true))

@show res
@show res.minimizer

# end
#
# end # module
