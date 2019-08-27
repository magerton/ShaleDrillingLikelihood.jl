# module ShaleDrillingLikelihood_Drilling_Test
#
# using ShaleDrillingLikelihood
# using Test
using StatsFuns
using Distributions
# using Calculus
# using Optim
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

nobs = 10
β = [0.0, 1.0, -2.0]
k = length(β)

# exogenous variable
X = randn(nobs)

# choice utilities
u = β*X'

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
ubV = zeros(k, nobs)
mul!(ubV, theta, X')


@inbounds @simd for d in drng
    ubv[d+1] = flow(u,d,obs) + β * dynamic
end



# end
#
# end # module
