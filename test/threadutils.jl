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
    # add1batch!,
    getrange,
    default_batch_size
    # batch,
    # batch_size,
    # len,
    # batchrange,
    # next!

function hh(x)
    n = length(x)
    mapper = Mapper2(n)
    ld = mapper.len
    atomic = mapper.atomic
    batch_size = mapper.batch_size # default_batch_size(n)

    s = Threads.Atomic{Float64}(0.0)
    Threads.@threads for j in 1:Threads.nthreads()
        while true
            k = ShaleDrillingLikelihood.add_iter!(mapper)
            batch_start = 1 + (k-1) * batch_size
            batch_end = min(k * batch_size, ld)
            batch_start > ld && break
            y = 0.0
            @inbounds @simd for i in batch_start:batch_end
                y += x[i]
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
            y += x[i]
        end
        Threads.atomic_add!(s, y)
    end
    s[]
end

function g(x)
    n = length(x)
    s = 0.0
    @inbounds @simd for i in 1:n
        s += x[i]
    end
    s
end

z = rand(10^8)
@testset "summation: getrange" begin
    println("--------")
    println("mapper")
    hh(z)
    @time _h = hh(z)
    println("threaded")
    f(z)
    @time _f = f(z)
    println("unthreaded")
    g(z)
    @time _g = g(z)
    @test _f ≈ _g
    @test _g ≈ _h
    @show _f, _g, _f - _g, _g - _h
end




end
