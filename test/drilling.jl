module ShaleDrillingLikelihood_Drilling_Test
#
# using ShaleDrillingLikelihood
using Test
# using Revise

# using Revise

# look at https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39
# https://discourse.julialang.org/t/two-questions-about-multithreading/14564/2
# https://discourse.julialang.org/t/question-about-multi-threading-performance/12075/3


using ShaleDrillingLikelihood
using StatsFuns
using Distributions
using Random
using BenchmarkTools
using Base.Threads
using Profile
using ProfileView
using InteractiveUtils


using Calculus
using Optim
using LinearAlgebra

# @testset "Drilling Likelihood" begin


# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

# -----------------------------------------------
# simulate data
# -----------------------------------------------

Random.seed!(1234)

num_i = 100
num_t = 50
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
M = 100*nthreads()
psisim = randn(M,num_i)


ubvtmp = Array{Float64}(undef, L+cache_pad, nthreads())
llmtmp = Array{Float64}(undef, M)
LLthread = zeros(Float64, nthreads())

yi, Xi, psii = choices[irng(num_t,1)], X[irng(num_t,1)], psisim[:,1]

f(g, θ) = -loglik(g, choices, X, psisim, θ, ubvtmp, llmtmp, num_t, num_i)
# @code_warntype loglik(loglik_i_thread, choices, X, psisim, theta, ubvtmp, llmtmp, num_t, num_i)

f(loglik_i_serial, theta)
f(loglik_i_thread, theta)

@show @benchmark  f(loglik_i_serial, theta)
@show @benchmark  f(loglik_i_thread, theta)


println("\n\n-------- Serial --------\n\n")
@benchmark     loglik_i_serial(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@code_warntype loglik_i_serial(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@profile       loglik_i_serial(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@benchmark   f(loglik_i_serial, theta)
@profile     f(loglik_i_serial, theta)
Profile.print()
ProfileView.view()

println("\n\n------- Threaded ---------\n\n")
@benchmark       loglik_i_thread(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@code_warntype   loglik_i_thread(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@profile         loglik_i_thread(yi, Xi, psii, theta, ubvtmp, llmtmp, L)
@profile       f(loglik_i_thread, theta)
Profile.print()
ProfileView.view()
#
# res = optimize(x -> f(loglik_i_serial, x), theta, BFGS(), Optim.Options(time_limit=60.0*5, show_trace=true))
# @show res
# @show res.minimizer
#
# res = optimize(x -> f(loglik_i_thread, x), theta, BFGS(), Optim.Options(time_limit=60.0*5, show_trace=true))
# @show res
# @show res.minimizer

# end
#
end # module
