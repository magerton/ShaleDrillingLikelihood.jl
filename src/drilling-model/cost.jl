export     AbstractDrillingCost,
    AbstractDrillingCost_TimeFE,
    DrillingCost_TimeFE,
    DrillingCost_TimeFE_rigrate,
    DrillingCost_constant,
    DrillingCost_dgt1

@inline flowdψ(::AbstractDrillingCost, d, obs, theta, sim) = azero(theta)
@inline flow!(grad, u::AbstractDrillingCost, d, obs, θ, sim)
    dflow!(u, grad, d, obs, θ, sim)
    return flow(u, d, obs, θ, sim)
end

# -------------------------------------------
# Drilling Cost
# -------------------------------------------

"Single drilling cost"
struct DrillingCost_constant <: AbstractDrillingCost end
@inline _nparm(x::DrillingCost_constant) = 1
@inline flow(::DrillingCost_constant, d, obs, θ, sim) = d*θ[1]
@inline function dflow!(::DrillingCost_constant, grad, d, obs, θ, sim)
    grad[1] = d
    return nothing
end

"Drilling cost changes if `d>1`"
struct DrillingCost_dgt1 <: AbstractDrillingCost end
@inline _nparm(x::DrillingCost_dgt1) = 2
@inline flow(::DrillingCost_dgt1, d, obs, θ, sim) = d*(d<=1 ? θ[1] : θ[2])
@inline function dflow!(::DrillingCost_dgt1, grad, d, obs, θ, sim)
    dgt1 = d>1
    grad[1] = !dgt1
    grad[2] = dgt1
    return nothing
end


"Abstract Type for Costs w Fixed Effects"
abstract type AbstractDrillingCost_TimeFE <: AbstractDrillingCost end
@inline start(    x::AbstractDrillingCost_TimeFE) = x.start
@inline stop(     x::AbstractDrillingCost_TimeFE) = x.stop
@inline startstop(x::AbstractDrillingCost_TimeFE) = start(x), stop(x)
@inline time_idx( x::AbstractDrillingCost_TimeFE, t::Integer) = clamp(t, start(x), stop(x)) - start(x) + 1
@inline time_idx(x, obs) = time_idx(x,year(obs))

"Time FE for 2008-2012"
struct DrillingCost_TimeFE <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE) = 2 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE, d, obs, θ, sim)
    d < 1 && return azero(θ)
    tidx = time_idx(u,obs)
    d == 1 && return θ[tidx]
    return d*(θ[tidx] + last(θ) )
end
@inline function dflow!(u::DrillingCost_TimeFE, grad, d, obs, θ, sim)
    fill!(grad, 0)
    if d > 0
        tidx = time_idx(u,obs)
        grad[tidx] += d
        if d > 1
            grad[end] += d
        end
    end
    return nothing
end


"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_costdiffs) = 4 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_costdiffs, d, obs, θ, sim)
    d < 1 && return azero(θ)
    tidx = time_idx(u,obs)
    r = θ[tidx]
    if !_Dgt0(obs)
        d > 1 && (r += θ[end-2])
    else
        r += θ[end-(d==1)]
    end
    return d*r
end

@inline function dflow!(u::DrillingCost_TimeFE_costdiffs, grad, d, obs, θ, sim)
    fill!(grad, 0)
    d < 1 && return nothing
    tidx = time_idx(u,obs)
    grad[tidx] = d
    if !_Dgt0(obs)
        d > 1 && ( grad[end-2] = d )
    else
        grad[end-(d==1)] = d
    end
    return nothing
end


"Time FE w rig rates for 2008-2012"
struct DrillingCost_TimeFE_rigrate <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_rigrate) = 3 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_rigrate, d, obs, θ, sim)
    d  < 1 && return azero(θ)
    tidx = time_idx(u,obs)
    r = θ[tidx] + θ[end]*rigrate(obs)
    d == 1 && return r
    return d*( r + θ[end-1])
end
@inline function dflow!(u::DrillingCost_TimeFE_rigrate, grad, d, obs, θ, sim)
    fill!(grad, 0)
    d == 0 && return nothing
    tidx = time_idx(u,obs)
    grad[tidx] = d
    d > 1 && (grad[end-1] = d)
    grad[end] = d*rigrate(obs)
    return nothing
end



"Time FE for 2008-2012 with shifters for (D==0,d>1), (D>1,d==1), (D>1,d>1)"
struct DrillingCost_TimeFE_rig_costdiffs <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline _nparm(x::DrillingCost_TimeFE_rig_costdiffs) = 5 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_rig_costdiffs, d, obs, θ, sim)
    d < 1 && return azero(θ)
    tidx = time_idx(u, obs)

    r = θ[tidx]
    if !_Dgt0(obs)
        d > 1  && (r += θ[end-3] )
    else
        r += θ[end-1-(d==1)]
    end
    return d*( r + θ[end]*rigrate(obs) )
end

@inline function dflow!(u::DrillingCost_TimeFE_rig_costdiffs, grad, d, obs, θ, sim)
    fill!(grad, 0)
    d == 0 && return nothing
    tidx = time_idx(u,obs)

    grad[tidx] += d
    if !_Dgt0(obs)
        d > 1 && ( grad[end-3] += d )
    else
        grad[end-1-(d==1)] += d
    end
    grad[end] += d*rigrate(obs)
    return nothing
end
