using ShaleDrillingLikelihood: zero_out_small_probs!

import ShaleDrillingLikelihood: logprice, logrigrate

using Statistics: var, cov

export GridTransition

# ----------------------------
# Make Grid + transition
# ----------------------------

mysteprange(a,b,n) = range(a,b;length=n)

function PriceGrid(z, delta, len)
    x = first.(z)
    prng = range(minimum(x)-delta, maximum(x)+delta; length=len)
    sigsq = var(diff(x))
    println("Var(logprice) = $sigsq and sd(logprice) = $(sqrt(sigsq))")
    P = tauchen_1d(prng, identity, sigsq)
    return prng, P
end

function PriceCostGrid(z, delta, len)
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

    Sigma = cov(dpc)
    stdvs = sqrt.(diag(Sigma))
    sigcor = Sigma[2,1] / prod(stdvs)
    println("Sigma(logprice, logrigrate) = $Sigma.\nStdv(logprice, logrigrate) = $(round.(stdvs;digits=5)) and correlation = $(round(sigcor;digits=4))")

    P = tauchen_2d(pcprod, identity, Sigma)
    return pcrng, P
end

function YearGrid(z, args...)
    y = ShaleDrillingLikelihood.year.(z)
    miny, maxy = extrema(y)
    yrng = UnitRange(miny, maxy)
    nyears = length(yrng)
    nyearsm1 = nyears-1
    @assert nyearsm1 == maxy-miny

    l = zeros(nyearsm1)
    m = vcat(fill(3/4,nyearsm1), 1.0)
    r = fill(1/4, nyearsm1)

    P = Tridiagonal(l,m,r)
    return yrng, P
end

function GridTransition(rwrd::DrillReward{R,C}, z::Vector, delta::Number, len::Number; minp=minp_default()) where {
        R, C<:Union{DrillingCost_constant, DrillingCost_dgt1}
    }
    rng, P = PriceGrid(z,delta,len)
    zero_out_small_probs!(P, minp)
    return tuple(rng), sparse(P)
end

# # rigrate + price
# function GridTransition(rwrd::DrillReward{R,C}, z, delta, len; minp=minp_default()) where {
#         R, C<:Union{DrillingCost_constant, DrillingCost_dgt1}
#     }
#     rng, P = PriceCostGrid(z, delta, len)
#     zero_out_small_probs!(P, minp)
#     return rng, sparse(P)
# end

function GridTransition(rwrd::DrillReward{R,C}, z::Vector, delta::Number, len::Number; minp=minp_default()) where {
        R, C<:AbstractDrillingCost_TimeFE
    }
    p = tuple.(logprice.(z))
    prng, pP = PriceGrid(p, delta, len)
    yrng, yP = YearGrid(z)
    rng = (prng, yrng)
    P = kron(yP, pP)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

function GridTransition(rwrd::DrillReward{R,C}, z::Vector, delta::Number, len::Number; minp=minp_default()) where {
        R, C<:Union{DrillingCost_TimeFE_rigrate, DrillingCost_TimeFE_rig_costdiffs}
    }
    pcrng, pcP = PriceCostGrid(z,delta,len)
    yrng, yP = YearGrid(z)
    rng = (pcrng..., yrng)
    P = kron(yP, pcP)
    zero_out_small_probs!(P, minp)
    return rng, sparse(P)
end

GridTransition(rwrd::DrillReward, etv::ExogTimeVars, args...; kwargs...) = GridTransition(rwrd, _timevars(etv), args...; kwargs...)
GridTransition(d::DataDrillPrimitive, args...; kwargs...) = GridTransition(reward(d), zchars(d), args...; kwargs...)
GridTransition(d::DataDrill,          args...; kwargs...) = zspace(_model(d)), ztransition(_model(d))
