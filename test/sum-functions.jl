module ShaleDrillingLikelihood_SumFunctions_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools

using ShaleDrillingLikelihood: mylogsumexp,
    logsumexp_and_softmax!,
    sumprod3,
    sumprod3_test

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
        println("$n elements")
        y = randn(n)
        x = view(y, 1:n)
        z1 = similar(y)
        z2 = similar(y)

        bmark = logsumexp(x)
        @test bmark ≈ mylogsumexp(x)
        @test bmark ≈ mylogsumexp(x,n)

        println("    logsumexp")
        @btime logsumexp($x)
        @btime mylogsumexp($x)
        @btime mylogsumexp($x,$n)

        softmax!(z1,y)
        logsumexp_and_softmax!(z2,y)
        @test sum(z2) ≈ 1
        @test z1 ≈ z2

        println("    softmax")
        @btime softmax!($z1,$y)
        @btime logsumexp_and_softmax!($z1,$y)
    end

end
end
