module ShaleDrillingLikelihood_tauchen

using ShaleDrillingLikelihood
using StatsFuns
using Test
using Statistics
using SparseArrays
using Random
using Base: product
using LinearAlgebra

using ShaleDrillingLikelihood: ar,
    simulate,
    mu,
    sigsq,
    condvar,
    lrvar,
    lrmean,
    MakeGrid,
    zero_out_small_probs

@testset "tauchen approximations to ar1" begin

    Random.seed!(1234)

    ar1 = AR1process(1.0, 0.6, 1.5)

    y = simulate(ar1, 100_000)

    @test mean(y) - lrmean(ar1) < 0.02
    @test var(y) - lrvar(ar1) < 0.02

    @test mean(y[2:end] .- ar(ar1).*y[1:end-1]) - mu(ar1) < 0.01
    @test var(y[2:end] .- ar(ar1).*y[1:end-1]) - condvar(ar1) < 0.01


    ngridpts = 21
    P = tauchen_1d(ar1, ngridpts)
    @test all(sum(P, dims=2) .≈ 1)
    xgrid = MakeGrid(ar1, ngridpts)

    num_outside = length(y) - sum(first(xgrid) .<= y .<= last(xgrid) )
    @test num_outside / length(y) - normcdf(-3.0)*2 < 0.0002

    Q = zero_out_small_probs(P, 1e-5)
    @test all(sum(Q, dims=2) .≈ 1)

    Qsp = sparse(Q)
    @test length(nonzeros(Qsp)) < length(Q)
end

@testset "random walk" begin
    rw = RandomWalk(1.0)
    y = simulate(rw, 1000)
    @test var(y) > var(y[1:100])
end


@testset "tauchen 2d tests" begin

    num_y_nodes = 21
    num_yy_nodes = num_y_nodes^2
    ylim = 3.0

    yspace = range(-abs(ylim); stop=abs(ylim), length=num_y_nodes)
    yyspace = product(yspace,yspace)
    Q = Matrix{Float64}(undef, num_y_nodes, num_y_nodes)
    P = Matrix{Float64}(undef, num_yy_nodes, num_yy_nodes)
    P2 = copy(P)

    μ(x) = x
    Σ = [1.0 0.5; 0.5 1.0]

    Q .= 0.0
    tauchen_1d!(Q, yspace, μ, 1.0)
    @test all(sum(Q, dims=2) .≈ 1.0)
    tauchen_2d!(P, yyspace, μ, Matrix{Float64}(1.0I,2,2))
    @test all(sum(P, dims=2) .≈ 1.0)

    @test P ≈ kron(Q,Q)


    tauchen_2d!(P, yyspace, μ, Σ)
    @test all(sum(P, dims=2) .≈ 1.0)
    P3 = copy(P)
    tauchen_2d!(P, product(yspace.*sqrt2, yspace.*sqrt2), μ, Σ.*2)
    @test P ≈ P3

    @test all(sum(P, dims=2) .≈ 1.0)
end








end # module
