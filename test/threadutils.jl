module ShaleDrillingLikelihood_ThreadUtils_Test

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads

using ShaleDrillingLikelihood: getrange,
    Mapper,
    Mapper2,
    AbstractThreadMapper,
    # add1batch!,
    getrange,
    default_batch_size,
    nextrange!
    # batch,
    # batch_size,
    # len,
    # batchrange,
    # next!

function h2(x)
    n = length(x)
    mapper = Mapper2(n)
    ld = ShaleDrillingLikelihood.stop(mapper)
    batch_size = ShaleDrillingLikelihood.step(mapper) # default_batch_size(n)

    s = Threads.Atomic{Float64}(0.0)
    Threads.@threads for j in 1:Threads.nthreads()
        while true
            k = ShaleDrillingLikelihood.add_iter!(mapper)
            batch_start = 1 + (k-1) * batch_size
            batch_end = min(k * batch_size, ld)
            batch_start > ld && break
            y = 0.0
            @inbounds @simd for i in batch_start:batch_end
                y += exp(x[i])
            end
            Threads.atomic_add!(s, y)
        end
    end
    return s[]
end



function h3(x)
    mapper = Mapper(length(x))
    s = Threads.Atomic{Float64}(0.0)

    Threads.@threads for j in 1:Threads.nthreads()
        while true
            y = 0.0
            irng = nextrange!(mapper)
            isnothing(irng) && break
            @inbounds @simd for i in irng
                y += exp(x[i])
            end
            Threads.atomic_add!(s, y)
        end
    end

    return s[]
end

function f(x)
    s = Threads.Atomic{Float64}(0.0)
    n = length(x)
    @Threads.threads for k in 1:Threads.nthreads()
        y = 0.0
        @inbounds @simd for i in getrange(n)
            y += exp(x[i])
        end
        Threads.atomic_add!(s, y)
    end
    s[]
end


function g(x)
    n = length(x)
    s = 0.0
    @inbounds @simd for i in 1:n
        s += exp(x[i])
    end
    s
end

gg(x) = mapreduce(exp, +, x)

z = rand(10^3)
@testset "summation: getrange" begin
    println("--------")

    println("Mapper")
    _h3 = h3(z)
    @btime h3($z)

    println("Mapper2")
    _h2 = h2(z)
    @btime h2($z)

    println("getrange")
    _f = f(z)
    @btime f($z)

    println("unthreaded")
    @btime g($z)
    _g = g(z)

    println("mapreduce")
    @btime gg($z)

    @test _f ≈ _g
    @test _g ≈ _h2
    @test _g ≈ _h3
end




end
