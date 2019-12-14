export     AbstractDrillingCost,
    AbstractDrillingCost_TimeFE,
    DrillingCost_TimeFE,
    DrillingCost_TimeFE_rigrate,
    DrillingCost_constant,
    DrillingCost_dgt1

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
@inline start(    x::AbstractDrillingCost_TimeFE) = x.start
@inline stop(     x::AbstractDrillingCost_TimeFE) = x.stop
@inline startstop(x::AbstractDrillingCost_TimeFE) = start(x), stop(x)
@inline time_idx( x::AbstractDrillingCost_TimeFE, t::Integer) = clamp(t, start(x), stop(x)) - start(x) + 1
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

"Time FE for 2008-2012"
struct DrillingCost_TimeFE <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE) = 2 + stop(x) - start(x)
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


"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_costdiffs) = 4 + stop(x) - start(x)
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



"Time FE w rig rates for 2008-2012"
struct DrillingCost_TimeFE_rigrate <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_rigrate) = 3 + stop(x) - start(x)
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


"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_rig_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_rig_costdiffs) = 5 + stop(x) - start(x)
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
