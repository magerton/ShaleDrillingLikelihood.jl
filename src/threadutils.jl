"""
    getrange(n)

Partition the range `1:n` into `Threads.nthreads()` subranges and return the one corresponding to `Threads.threadid()`.
Useful for splitting a workload among multiple threads. See also the `TiledIteration` package for more advanced variants.
"""
function getrange(n)
    tid = Threads.threadid()
    nt = Threads.nthreads()
    d , r = divrem(n, nt)
    from = (tid - 1) * d + min(r, tid - 1) + 1
    to = from + d - 1 + (tid ≤ r ? 1 : 0)
    from:to
end

# kissthreading
function kissf(x)
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

default_batch_size(n) = min(n, round(Int, 10*sqrt(n)))

struct Mapper
    atomic::Threads.Atomic{Int}
    len::Int
end

@inline function (mapper::Mapper)(batch_size, f, dst, src...)
    ld = mapper.len
    atomic = mapper.atomic
    Threads.@threads for j in 1:Threads.nthreads()
        while true
            k = Threads.atomic_add!(atomic, 1)
            batch_start = 1 + (k-1) * batch_size
            batch_end = min(k * batch_size, ld)
            batch_start > ld && break
            batch_map!(batch_start:batch_end, f, dst, src...)
        end
    end
    dst
end


# struct Mapper
#     batch::Atomic{Int}
#     len::Int
#     batch_size::Int
#     function Mapper(len, batch_size=1)
#         batch = Atomic{Int}(1)
#         batch_size < len || throw(DomainError())
#         return new(batch, len, batch_size)
#     end
# end
#
# batch(m::Mapper) = m.batch
# batch_size(m::Mapper) = m.batch_size
# len(m::Mapper) = m.len
#
# add1batch!(m::Mapper) = atomic_add!(batch(m), 1)
# function batchrange(m::Mapper)
#     k = batch(m)[]
#     batch_start = 1 + (k-1) * batch_size(m)
#     batch_end = min(k * batch_size(m), len(m))
#     if batch_start > len(m)
#         return 1:0
#     else
#         return batch_start:batch_end
#     end
# end
#
# function next!(m::Mapper)
#     add1batch!(m)
#     return batchrange(m)
# end
