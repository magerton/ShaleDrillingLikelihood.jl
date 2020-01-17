module ShaleDrillingLikelihood_SumFunctions_Test

DOBTIME = false

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using InteractiveUtils

using StatsFuns: softmax!

using ShaleDrillingLikelihood: logsumexp!,
    sumprod3,
    softmax3test!,
    logsumexp3test!

using Base: OneTo

@testset "sumprod3" begin
    for n in (1,2,3,4,10,)
        x = randn(n)
        y = randn(n)
        z = randn(n)
        @test sum(x .* y .* z) ≈ sumprod3(x,y,z)
    end
end


@testset "Custom logsumexp functions" begin

    for n in (1,2,3,4,10,100,1000,10_000)
        # println("test logsumexp! for $n elements")
        y = randn(n)
        x = view(y, 1:n)
        z1 = similar(y)
        z2 = similar(y)

        bmark = logsumexp(x)
        @test bmark ≈ logsumexp!(z1,x)
        # @show @code_warntype logsumexp!(z1,x)

        fill!(z1,0)
        softmax!(z1,y)
        logsumexp!(z2,y)
        @test sum(z2) ≈ 1
        @test z1 ≈ z2

        if DOBTIME
            @btime logsumexp($x)
            @btime softmax!($z1,$y)
            @btime logsumexp!($z1,$y)
        end
    end

end

println("printme")

@testset "softmax3! functionw" begin

    ni, nj, nk = (100, 100, 10)
    x0 = rand(ni, nj, nk)
    q0 = similar(x0)

    # x = view(x0, :, :, OneTo(nk))
    # q = view(q0, :, :, OneTo(nk))
    x = x0
    q = q0

    tmpmax = zeros(ni,nj)
    lse = similar(tmpmax)

    qcopy = similar(q)
    softmax3test!(qcopy, x)
    @test sum(qcopy; dims=3) ≈ ones(ni,nj)

    lsetest = logsumexp3test!(x)

    # test softmax3 broadcast
    # @show @code_warntype softmax3!(q, lse, tmpmax, x, 1)
    softmax3!(q, lse, tmpmax, x)
    @test sum(q; dims=3) ≈ ones(ni,nj)
    @test q ≈ qcopy

    # test w/ view
    fill!(q,0)
    fill!(lse,0)
    @views qvw = q[:,:,1:end]
    @views xvw = x[:,:,1:end]
    # @show @code_warntype softmax3!(qvw, lse, tmpmax, xvw, 1)
    softmax3!(qvw, lse, tmpmax, xvw, 1)

    # test logsumexp
    fill!(q,0)
    fill!(lse,0)
    logsumexp_and_softmax!(lse, q, tmpmax, x, 1)
    @test lse ≈ lsetest
    @test all(q[:,:,1] .< 1)
    @test q[:,:,1] ≈ qcopy[:,:,1]
    @test !(q ≈ qcopy)

    if DOBTIME
        @btime softmax3!($q, $lse, $tmpmax, $x)
        @btime softmax3test!($q, $x)
    end


end


end
