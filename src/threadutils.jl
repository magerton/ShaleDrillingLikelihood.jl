abstract type AbstractThreadMapper end

default_batch_size(n) = default_step(n)
default_step(n) =  min(n, round(Int, 10*sqrt(n)))

function getrange(n)
    tid = Threads.threadid()
    nt = Threads.nthreads()
    d , r = divrem(n, nt)
    from = (tid - 1) * d + min(r, tid - 1) + 1
    to = from + d - 1 + (tid ≤ r ? 1 : 0)
    from:to
end

struct Mapper <: AbstractThreadMapper
    start::Threads.Atomic{Int}
    stop::Int
    step::Int
    function Mapper(n::Int, step::Int=default_step(n))
        0 < n < typemax(Int) || throw(DomainError())
        start = Threads.Atomic{Int}(0)
        stop = n
        return new(start, stop, step)
    end
end

struct Mapper2 <: AbstractThreadMapper
    iter::Threads.Atomic{Int}
    stop::Int
    step::Int
    function Mapper2(stop::Int, step::Int=default_step(stop))
        iter = Threads.Atomic{Int}(1)
        return new(iter, stop, step)
    end
end


iter(m::Mapper2) = m.iter
start(m::Mapper) = m.start
stop(m::AbstractThreadMapper) = m.stop
step(m::AbstractThreadMapper) = m.step

@deprecate batch_size(m::AbstractThreadMapper) step(m)
@deprecate atomic(m::AbstractThreadMapper) start(m)
@deprecate len(m::AbstractThreadMapper) stop(m)

add_iter!(m::Mapper2) = Threads.atomic_add!(iter(m),1)

function nextrange!(m::Mapper)
    strt = Threads.atomic_add!(start(m), step(m))
    if strt >= stop(m)
        return nothing
    else
        a = strt+1
        b = min(strt + step(m), stop(m))
        return a:b
    end
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
