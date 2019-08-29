module ShaleDrillingLikelihood_Drilling_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads
# using Profile
# using ProfileView
# using InteractiveUtils

using Calculus
using Optim
using LinearAlgebra

@testset "Drilling Likelihood" begin

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
    M = 10*nthreads()
    psisim = randn(M,num_i)

    @test true == true

    println("\n\nmade simulations... run serial once")
    llserl = loglik_serial(choices, X, psisim, theta, num_t, num_i)
    println("run threaded once")
    llthrd = loglik_threaded(choices, X, psisim, theta, num_t, num_i)
    @test llserl == llthrd

    # # @code_warntype loglik_serial(choices, X, psisim, theta, num_t, num_i)
    # # @code_warntype loglik_threaded(choices, X, psisim, theta, num_t, num_i)

    println("\n\n-------- Serial --------\n\n")
    @show @btime loglik_serial($choices, $X, $psisim, $theta, $num_t, $num_i)

    println("\n\n-------- Threaded using $(nthreads()) threads --------\n\n")
    @show @btime loglik_threaded($choices, $X, $psisim, $theta, $num_t, $num_i)

    println("\n\n-------- older benchmarks --------\n\n")
    ubvtmp = Array{Float64}(undef, L+cache_pad, nthreads())
    llmtmp = Array{Float64}(undef, M)
    f(g, θ) = loglik(g, choices, X, psisim, θ, ubvtmp, llmtmp, num_t, num_i)
    llserl_old = f(loglik_i_serial, theta)
    llthrd_old = f(loglik_i_thread, theta)
    @test llserl_old == llserl
    @test llthrd_old == llserl

    println("\nSerial:")
    @show @btime  loglik(loglik_i_serial, $choices, $X, $psisim, $theta, $ubvtmp, $llmtmp, $num_t, $num_i)
    println("\nThreaded:")
    @show @btime  loglik(loglik_i_thread, $choices, $X, $psisim, $theta, $ubvtmp, $llmtmp, $num_t, $num_i)

    @test true == true

end # testset
end # module
