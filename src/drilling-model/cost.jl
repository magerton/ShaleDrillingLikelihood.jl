export     AbstractDrillingCost,
    AbstractDrillingCost_TimeFE,
    DrillingCost_TimeFE,
    DrillingCost_TimeFE_rigrate,
    DrillingCost_constant,
    DrillingCost_dgt1,
    DrillingCost_TimeFE_costdiffs,
    DrillingCost_TimeFE_rig_costdiffs


@inline flowdψ(::AbstractDrillingCost, d, obs, theta, sim) = azero(theta)
@inline function flow!(grad, u::AbstractPayoffFunction, d, obs, θ, sim)
    dflow!(u, grad, d, obs, θ, sim)
    return flow(u, d, obs, θ, sim)
end

@inline function flow!(u::AbstractPayoffFunction, grad, d, obs, θ, sim)
    dflow!(u, grad, d, obs, θ, sim)
    return flow(u, d, obs, θ, sim)
end



"Abstract Type for Costs w Fixed Effects"
abstract type AbstractDrillingCost_TimeFE <: AbstractDrillingCost end
(x::Type{T})(a, b) where {T<:AbstractDrillingCost_TimeFE} = T(UnitRange(Float64(a), Float64(b)))
yearrange(x::AbstractDrillingCost_TimeFE) = x.yearrange
@inline start(x::AbstractDrillingCost_TimeFE) = first(yearrange(x))
@inline stop(x::AbstractDrillingCost_TimeFE) = last(yearrange(x))
@inline startstop(x::AbstractDrillingCost_TimeFE) = start(x), stop(x)
@inline time_idx( x::AbstractDrillingCost_TimeFE, t::Number) = Int(clamp(t, start(x), stop(x)) - start(x) + 1)
@inline time_idx(x, obs) = time_idx(x,year(obs))

# -------------------------------------------
# Drilling Cost
# -------------------------------------------

"Single drilling cost"
struct DrillingCost_constant <: AbstractDrillingCost end
@inline _nparm(x::DrillingCost_constant) = 1
@inline function flow!(grad, ::DrillingCost_constant, d, obs, θ, sim, dograd::Bool)
    if dograd
        grad[1] = d
    end
    return d*θ[1]
end
coefnames(::DrillingCost_constant) = ["\\alpha_c",]



"Drilling cost changes if `d>1`"
struct DrillingCost_dgt1 <: AbstractDrillingCost end
@inline _nparm(x::DrillingCost_dgt1) = 2
@inline function flow!(grad, ::DrillingCost_dgt1, d, obs, θ, sim, dograd::Bool)
    T = eltype(θ)
    if d == 0
        dograd && fill!(grad, 0)
        u = zero(T)
    else
        dgt1 = d>1
        u = d*θ[1+dgt1]
        if dograd
            grad[1+dgt1] = d
            grad[2-dgt1] = 0
        end
    end
    return u::T
end
coefnames(::DrillingCost_dgt1) = ["\\alpha_{d=1}", "\\alpha_{d>1}"]



"Time FE for 2008-2012"
struct DrillingCost_TimeFE <: AbstractDrillingCost_TimeFE
    yearrange::UnitRange{Float64}
    # start::Float64
    # stop::Float64
end
@inline _nparm(x::DrillingCost_TimeFE) = 1 + length(yearrange(x))
@inline function flow!(grad, u::DrillingCost_TimeFE, d, obs, θ, sim, dograd::Bool)
    T = eltype(θ)
    dograd && fill!(grad, 0)
    d == 0 && return zero(T)

    tidx = time_idx(u,obs)
    dgt1 = d > 1

    r = d*(θ[tidx] + dgt1*θ[end])
    if dograd
        grad[tidx] = d
        grad[end]  = d*dgt1
    end
    return r::T
end
function coefnames(x::DrillingCost_TimeFE)
    cfs = ["\\alpha_{$y}" for y in start(x):stop(x)]
    return vcat(cfs, "\\alpha_{d>1}")
end




"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_costdiffs <: AbstractDrillingCost_TimeFE
    yearrange::UnitRange{Float64}
    # start::Float64
    # stop::Float64
end
@inline _nparm(x::DrillingCost_TimeFE_costdiffs) = 3 + length(yearrange(x))
@inline function flow!(grad, u::DrillingCost_TimeFE_costdiffs, d, obs, θ, sim, dograd::Bool)
    T = eltype(θ)
    dograd && fill!(grad, 0)
    d == 0 && return zero(T)

    tidx = time_idx(u,obs)
    Dgt0 = _Dgt0(obs)
    dgt1 = d > 1

    r = d * ( θ[tidx] + (!Dgt0*dgt1)*θ[end-2] + Dgt0*θ[end-1+dgt1] )
    if dograd
        grad[tidx]       = d
        grad[end-2]      = d * !Dgt0*dgt1
        grad[end-1+dgt1] = d * Dgt0
    end
    return r::T
end

function coefnames(x::DrillingCost_TimeFE_costdiffs)
    cfs = ["\\alpha_{$y}" for y in start(x):stop(x)]
    return vcat(cfs, "\\alpha_{D=0,d>1}", "\\alpha_{D>0,d=1}", "\\alpha_{D>0,d>1}")
end




"Time FE w rig rates for 2008-2012"
struct DrillingCost_TimeFE_rigrate <: AbstractDrillingCost_TimeFE
    yearrange::UnitRange{Float64}
    # start::Float64
    # stop::Float64
end
@inline _nparm(x::DrillingCost_TimeFE_rigrate) = 2 + length(yearrange(x))
@inline function flow!(grad, u::DrillingCost_TimeFE_rigrate, d, obs, θ, sim, dograd::Bool)
    T = eltype(θ)
    dograd && fill!(grad, 0)
    d == 0 && return zero(T)

    tidx = time_idx(u,obs)
    rig = rigrate(obs)
    Dgt0 = _Dgt0(obs)
    dgt1 = d > 1

    r = d*(θ[tidx] + dgt1*θ[end-1] + θ[end]*rig)
    if dograd
        grad[tidx]  = d
        grad[end-1] = d*dgt1
        grad[end]   = d*rig
    end
    return r
end

function coefnames(x::DrillingCost_TimeFE_rigrate)
    cfs = ["\\alpha_{$y}" for y in start(x):stop(x)]
    return vcat(cfs, "\\alpha_{d>1}", "\\alpha_{rig}")
end




"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_rig_costdiffs <: AbstractDrillingCost_TimeFE
    yearrange::UnitRange{Float64}
    # start::Float64
    # stop::Float64
end
@inline _nparm(x::DrillingCost_TimeFE_rig_costdiffs) = 4 + length(yearrange(x))
@inline function flow!(grad, u::DrillingCost_TimeFE_rig_costdiffs, d, obs, θ, sim, dograd::Bool)
    dograd && fill!(grad, 0)
    d == 0 && return zero(T)

    tidx = time_idx(u,obs)
    rig = rigrate(obs)
    Dgt0 = _Dgt0(obs)
    dgt1 = d > 1

    r = d * ( θ[tidx] + (!Dgt0*dgt1)*θ[end-3] + Dgt0*θ[end-2+dgt1] + rig)
    if dograd
        grad[tidx]       = d
        grad[end-3]      = d * (!Dgt0*dgt1)
        grad[end-2+dgt1] = d * Dgt0
        grad[end]        = d * rig
    end
    return r
end

function coefnames(x::DrillingCost_TimeFE_rig_costdiffs)
    cfs = ["\\alpha_{$y}" for y in start(x):stop(x)]
    return vcat(cfs, "\\alpha_{D=0,d>1}", "\\alpha_{D>0,d=1}", "\\alpha_{D>0,d>1}", "\\alpha_{rig}")
end
