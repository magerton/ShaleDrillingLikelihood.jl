using ShaleDrillingLikelihood: PriceTuple,
    PriceCostTuple,
    PriceCostYearTuple,
    PriceYearTuple,
    zero_out_small_probs!

import ShaleDrillingLikelihood: logprice, logrigrate

using Statistics: var, cov

export GridTransition

# ----------------------------
# Make Grid + transition
# ----------------------------

mysteprange(a,b,n) = range(a,b;length=n)

function PriceGrid(z::Vector{PriceTuple}, delta, len)
    x = first.(z)
    prng = range(minimum(x)-delta, maximum(x)+delta; length=len)
    sigsq = var(diff(x))
    P = tauchen_1d(prng, identity, sigsq)
    return tuple(prng), P
end

function PriceCostGrid(z::Vector{T}, delta, len) where {T<:Union{PriceCostTuple, PriceCostYearTuple}}
    p = logprice.(z)
    c = logrigrate.(z)
    pc = hcat(p,c)

    # ranges
    mins = vec(minimum(pc; dims=1))
    maxs = vec(maximum(pc; dims=1))
    a = mins .- delta
    b = maxs .+ delta
    pcrng = mysteprange.(a, b, len)
    pcprod = product(pcrng...)

    # sigma
    dpc = diff(pc; dims=1)

    P = tauchen_2d(pcprod, identity, cov(dpc))
    return pcrng, P
end

function YearGrid(z::Vector{T}, args...) where {T<:Union{PriceYearTuple, PriceCostYearTuple}}
    y = ShaleDrillingLikelihood.year.(z)
    miny, maxy = extrema(y)
    yrng = miny : maxy
    nyears = length(yrng)
    nyearsm1 = nyears-1

    l = zeros(nyearsm1)
    m = vcat(fill(3/4,nyearsm1), 1.0)
    r = fill(1/4, nyearsm1)

    P = Tridiagonal(l,m,r)
    return yrng, P
end

function GridTransition(z::Vector{<:PriceTuple}, delta, len; minp=minp_default())
    rng, P = PriceGrid(z,delta,len)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

function GridTransition(z::Vector{<:PriceCostTuple}, delta, len; minp=minp_default())
    rng, P = PriceCostGrid(z, delta, len)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

function GridTransition(z::Vector{<:PriceYearTuple}, delta, len; minp=minp_default())
    p = tuple.(logprice.(z))
    prng, pP = PriceGrid(p, delta, len)
    yrng, yP = YearGrid(z)
    rng = (prng, yrng)
    P = kron(yP, pP)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

function GridTransition(z::Vector{<:PriceCostYearTuple}, delta, len; minp=minp_default())
    pcrng, pcP = PriceCostGrid(z,delta,len)
    yrng, yP = YearGrid(ShaleDrillingLikelihood.year.(z))
    rng = (pcrng..., yrng)
    P = kron(yP, pcP)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

GridTransition(etv::ExogTimeVars, args...; kwargs...) = GridTransition(_timevars(etv), args...; kwargs...)

GridTransition(d::DataDrillPrimitive, args...; kwargs...) = GridTransition(zchars(d), args...; kwargs...)
GridTransition(d::DataDrill,          args...; kwargs...) = zspace(_model(d)), ztransition(_model(d))
