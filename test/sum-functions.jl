module ShaleDrillingLikelihood_SumFunctions_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools

using ShaleDrillingLikelihood: logsumexp!,
    sumprod3


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
        println("test logsumexp! for $n elements")
        y = randn(n)
        x = view(y, 1:n)
        z1 = similar(y)
        z2 = similar(y)

        bmark = logsumexp(x)
        @test bmark ≈ logsumexp!(z1,x)

        fill!(z1,0)
        softmax!(z1,y)
        logsumexp!(z2,y)
        @test sum(z2) ≈ 1
        @test z1 ≈ z2

        # @btime logsumexp($x)
        # @btime softmax!($z1,$y)
        # @btime logsumexp!($z1,$y)
    end

end
end
