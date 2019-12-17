export tauchen_2d!,
    tauchen_1d!,
    tauchen_2d,
    tauchen_1d

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


minp_default() = 1e-5

function zero_out_small_probs!(Q::AbstractMatrix, P::AbstractMatrix, minp::Real)
    checksquare(P) == checksquare(Q) || throw(DimensionMismatch())
    @assert all(sum(P, dims=2) .≈ 1)
    for i in eachindex(P)
        p = P[i]
        Q[i] = p > minp ? p : 0
    end
    Q ./= sum(Q, dims=2)
    return Q
end

zero_out_small_probs!(P,minp) = zero_out_small_probs!(P,P,minp)
zero_out_small_probs( P,minp) = zero_out_small_probs!(similar(P), P, minp)
