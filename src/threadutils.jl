# https://stackoverflow.com/questions/58580436/how-to-make-use-of-threads-optional-in-a-julia-function
# https://slides.com/valentinchuravy/julia-parallelism#/

default_step(n) =  min(n, round(Int, 10*sqrt(n)))

abstract type AbstractThreadMapper end

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

stop(m::AbstractThreadMapper) = m.stop
step(m::AbstractThreadMapper) = m.step

start(m::Mapper) = m.start

iter(m::Mapper2) = m.iter
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
