export tauchen_2d!, tauchen_1d!,
    tauchen_2d, tauchen_1d,
    AR1process,
    approxgrid,
    TimeSeriesProcess,
    RandomWalkProcess

function bvn_upperlower_cdf(xlim, ylim, r)
    xl,xu = xlim
    yl,yu = ylim
    xl < xu && yl < yu || throw(DomainError())
    return bvncdf(xu,yu,r) - bvncdf(xl,yu,r) - bvncdf(xu,yl,r) + bvncdf(xl,yl,r)
end

function plus_minus_dx(x::Number, xspace::AbstractRange, scaling::Number=1)
  dx = step(xspace)/2
  x == first(xspace) && return (-Inf, dx,).*scaling
  x == last(xspace)  && return (-dx, Inf,).*scaling
  return (-dx, dx).*scaling
end

x_pm_Δ(x::Number, xspace::AbstractVector) = x .+ plus_minus_dx(x,xspace)

function tauchen_1d!(P::AbstractMatrix, S::AbstractVector, μ::Function, σ2::Number)
    σ2 > 0 || throw(DomainError())
    all(length(S) .== size(P)) || throw(DimensionMismatch())

    sigma = sqrt(σ2)
    invsigma = inv(sigma)

    for (j,s1) in enumerate(S)
      pm_dz = plus_minus_dx(s1, S, invsigma)
      for (i,s0) in enumerate(S)
        zm, zp = ((s1 - μ(s0))/sigma) .+ pm_dz
        P[i,j] = normcdf(zp) - normcdf(zm)
      end
    end
end

function tauchen_1d(S::AbstractVector{T}, μ::Function, σ2::Number) where {T<:Real}
    n = length(S)
    P = Matrix{T}(undef,n,n)
    tauchen_1d!(P,S,μ,σ2)
    return P
end

function tauchen_2d(S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix{T}) where {T<:Real}
  n = length(S)
  P = Matrix{T}(undef,n,n)
  tauchen_2d!(P,S,μ,Σ)
  return P
end

function tauchen_2d!(P::AbstractMatrix, S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix)
    ndims(S) == 2
    2 == checksquare(Σ) || throw(DimensionMismatch("Σ must be 2x2"))
    issymmetric(Σ) || throw(error("Σ must be symmetric"))
    length(S) == checksquare(P) || throw(DimensionMismatch())

    sigmas = sqrt.(diag(Σ))
    invsigmas = inv.(sigmas)
    ρ = Σ[1,2] / prod(sigmas)

    for (j,s1) in enumerate(S)
      pm_dz = plus_minus_dx.(s1, S.iterators, invsigmas)
      for (i,s0) in enumerate(S)
        z = (s1 .- μ(s0))./sigmas
        P[i,j] = bvn_upperlower_cdf(z[1] .+ pm_dz[1], z[2] .+ pm_dz[2], ρ)
      end
    end
end

abstract type TimeSeriesProcess end
abstract type AbstractAR1Process{T<:Real} end

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


struct RandomWalkProcess{T} <: AbstractAR1Process{T}
    sigsq::T
    function RandomWalkProcess(sigsq::T) where {T}
        sigsq > 0 || throw(DomainError(sigsq))
        return new{T}(sigsq)
    end
end

ar(x::RandomWalkProcess) = 1
mu(x::RandomWalkProcess) = 0

function approxgrid(x::AR1process, n; m=3)
    lrstdev = sqrt(lrvar(x))
    centered = range(-m*lrstdev; stop=m*lrstdev, length=n)
    return lrmean(x) .+ centered
end

function tauchen_1d(x::AR1process, n; kwargs...)
    xgrid = approxgrid(x, n; kwargs...)
    return tauchen_1d(xgrid, xt -> condmean(x,xt), condvar(x))
end

starting_dist(ar1::AR1process) = Normal(lrmean(ar1), lrstd(ar1))
starting_dist(ar1::RandomWalkProcess) = Normal(0, condstd(ar1))

function simulate(ar1::AbstractAR1Process{T}, n::Integer) where {T}
    dist_uncond = starting_dist(ar1)
    dist_cond   = Normal(zero(T), condstd(ar1))

    y = Vector{T}(undef,n)
    y[1] = rand(dist_uncond)
    Distributions.rand!(dist_cond, view(y, 2:n))
    for t in 2:n
        y[t] += condmean(ar1, y[t-1])
    end
    return y
end



function zero_out_small_probs(P::AbstractMatrix, minp::Real)
    checksquare(P)
    @assert all(sum(P, dims=2) .≈ 1)

    Q = deepcopy(P)
    for i in eachindex(Q)
        if Q[i] < minp
            Q[i] = 0
        end
    end
    Q ./= sum(Q, dims=2)
    return Q
end
