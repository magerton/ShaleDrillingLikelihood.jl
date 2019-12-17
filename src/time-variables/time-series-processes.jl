export AR1process,
    TimeSeriesProcess,
    RandomWalk,
    AbstractAR1Process

abstract type TimeSeriesProcess end
abstract type AbstractAR1Process{T<:Real} <: TimeSeriesProcess end

# ----------------------------
# AR1
# ----------------------------

struct AR1process{T<:Real} <: AbstractAR1Process{T}
    mu::T
    ar::T
    sigsq::T
    function AR1process(mu::T,ar::T,sigsq::T) where {T}
        -1 < ar < 1 || @warn("nonstationary: |ar| parameter >= 1")
        sigsq > 0 || throw(DomainError(sigsq))
        return new{T}(mu,ar,sigsq)
    end
end

mu(x::AR1process) = x.mu
ar(x::AR1process) = x.ar
sigsq(x::AbstractAR1Process) = x.sigsq
condvar(x::AbstractAR1Process) = sigsq(x)
condstd(x::AbstractAR1Process) = sqrt(condvar(x))
lrvar(x::AR1process) = sigsq(x) / (1 - ar(x)^2)
lrstd(x::AR1process) = sqrt(lrvar(x))
condmean(x::AbstractAR1Process, yt) = mu(x) + ar(x)*yt
lrmean(x::AR1process) = mu(x)/(1-ar(x))

# ----------------------------
# Random walk
# ----------------------------

struct RandomWalk{T} <: AbstractAR1Process{T}
    sigsq::T
    function RandomWalk(sigsq::T) where {T}
        sigsq > 0 || throw(DomainError(sigsq))
        return new{T}(sigsq)
    end
end

ar(x::RandomWalk) = 1
mu(x::RandomWalk) = 0

# ----------------------------
# Bivariate Brownian
# ----------------------------

function tauchen_1d(x::AbstractAR1Process, xgrid::AbstractRange)
    return tauchen_1d(xgrid, xt -> condmean(x,xt), condvar(x))
end

function tauchen_1d(x::AR1process, n::Integer; kwargs...)
    xgrid = MakeGrid(x, n; kwargs...)
    return tauchen_1d(xgrid, xt -> condmean(x,xt), condvar(x))
end

# ----------------------------
# Make Grid only
# ----------------------------

function MakeGrid(x::AR1process, n::Real; m=3)
    lrstdev = sqrt(lrvar(x))
    centered = range(-m*lrstdev; stop=m*lrstdev, length=n)
    return lrmean(x) .+ centered
end

function MakeGrid(x::AR1process, n::Real, z::AbstractVector; m=3)
    rng = MakeGrid(x,n)
    xz = (x,z)
    len = length(rng)
    a = minimum(minimum.(xz))
    b = maximum(maximum.(xz))
    return range(a; stop=b, length=len)
end

function MakeGrid(p::AR1process, zvec, length=15; process=TestPriceProcess())
    zmin = min(minimum(zvec), lrmean(process)-3*lrstd(process))
    zmax = max(maximum(zvec), lrmean(process)+3*lrstd(process))
    zrng = range(zmin, zmax; length=length)
    return tuple(zrng)
end

@deprecate approxgrid(args...) MakeGrid(args...)

# ----------------------------
# simulations
# ----------------------------

starting_dist(ar1::AR1process) = Normal(lrmean(ar1), lrstd(ar1))
starting_dist(ar1::RandomWalk) = Normal(0, condstd(ar1))

function simulate(ar1::AbstractAR1Process{T}, y0::Real, n::Integer) where {T}
    dist_cond   = Normal(zero(T), condstd(ar1))
    y = Vector{T}(undef,n)
    y[1] = y0
    Distributions.rand!(dist_cond, view(y, 2:n))
    for t in 2:n
        y[t] += condmean(ar1, y[t-1])
    end
    return y
end

function simulate(ar1::AbstractAR1Process, n::Integer)
    dist_uncond = starting_dist(ar1)
    y0 = rand(dist_uncond)
    simulate(ar1, y0, n)
end
