module ShaleDrillingLikelihood_ThreadUtils_Test

DOBTIME = false

using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads

using Base.Threads: nthreads, @threads, Atomic, atomic_add!

using ShaleDrillingLikelihood: getrange,
    Mapper,
    Mapper2,
    AbstractThreadMapper,
    getrange,
    nextrange!,
    stop,
    step,
    add_iter!


function h2(x)
    n = length(x)
    mapper = Mapper2(n)
    ld = stop(mapper)
    batch_size = step(mapper)

    s = Atomic{Float64}(0.0)
    @threads for j in 1:nthreads()
        while true
            k = add_iter!(mapper)
            batch_start = 1 + (k-1) * batch_size
            batch_end = min(k * batch_size, ld)
            batch_start > ld && break
            y = 0.0
            @inbounds @simd for i in batch_start:batch_end
                y += exp(x[i])
            end
            atomic_add!(s, y)
        end
    end
    return s[]
end



function h3(x)
    mapper = Mapper(length(x))
    s = Atomic{Float64}(0.0)

    @threads for j in 1:nthreads()
        while true
            y = 0.0
            irng = nextrange!(mapper)
            isnothing(irng) && break
            @inbounds @simd for i in irng
                y += exp(x[i])
            end
            atomic_add!(s, y)
        end
    end

    return s[]
end

function f(x)
    s = Atomic{Float64}(0.0)
    n = length(x)
    @threads for k in 1:nthreads()
        y = 0.0
        @inbounds @simd for i in getrange(n)
            y += exp(x[i])
        end
        atomic_add!(s, y)
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

function ggg(x)
    n = length(x)
    s = 0.0
    for i in 1:n
        s += exp(x[i])
    end
    s
end


z = rand(10^3)
@testset "summation: getrange" begin
    println("--------")

    println("Mapper")
    _h3 = h3(z)
    DOBTIME && @btime h3($z)

    println("Mapper2")
    _h2 = h2(z)
    DOBTIME && @btime h2($z)

    println("getrange")
    _f = f(z)
    DOBTIME && @btime f($z)

    println("unthreaded")
    DOBTIME && @btime g($z)
    _g = g(z)

    println("mapreduce")
    DOBTIME && @btime gg($z)
    
    println("vanilla")
    DOBTIME && @btime ggg($z)

    @test _f ≈ _g
    @test _g ≈ _h2
    @test _g ≈ _h3
end




end
