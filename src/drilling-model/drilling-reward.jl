"Reward function for basic drilling model"
struct DrillReward{R<:AbstractDrillingRevenue,C<:AbstractDrillingCost,E<:AbstractExtensionCost} <: AbstractStaticPayoff
    revenue::R
    drill::C
    extend::E
end

# access components
revenue(x::DrillReward) = x.revenue
drill(  x::DrillReward) = x.drill
extend( x::DrillReward) = x.extend
cost(   x::DrillReward) = drill(x)

# -----------------------------------------
# lengths
# -----------------------------------------

_nparms(    x::DrillReward) = (_nparm(cost(x)), _nparm(extend(x)), _nparm(revenue(x)))
_nparm(     x::DrillReward) = sum(_nparms(x))
idx_cost(   x::DrillReward) = OneTo(_nparm(cost(x)))
idx_extend( x::DrillReward) = OneTo(_nparm(extend(x)))  .+  _nparm(cost(x))
idx_revenue(x::DrillReward) = OneTo(_nparm(revenue(x))) .+ (_nparm(cost(x)) + _nparm(extend(x)))
idx_ρ(      x::DrillReward) = _nparm(x) # idx_ρ(revenue(x), idx_revenue(x)

idx_drill_ρ(x::DrillReward) = idx_ρ(x)

vw_cost(   x::DrillReward, theta) = view(theta, idx_cost(x))
vw_extend( x::DrillReward, theta) = view(theta, idx_extend(x))
vw_revenue(x::DrillReward, theta) = view(theta, idx_revenue(x))

# -----------------------------------------
# flow payoffs
# -----------------------------------------

function flow!(grad, x::DrillReward, d, obs, θ, sim, dograd)
    e = flow!(vw_extend( x, grad), extend( x), d, obs, vw_extend( x, θ), sim, dograd)
    c = flow!(vw_cost(   x, grad), cost(   x), d, obs, vw_cost(   x, θ), sim, dograd)
    r = flow!(vw_revenue(x, grad), revenue(x), d, obs, vw_revenue(x, θ), sim, dograd)
    return e+c+r
end

function flowdψ(x::DrillReward, d, obs, θ, sim)
    T = eltype(θ)
    if d == 0
        u = flowdψ(extend(x), d, obs, vw_extend(x,θ), sim)
    else
        c = flowdψ(cost(x),    d, obs, vw_cost(x,θ), sim)
        r = flowdψ(revenue(x), d, obs, vw_revenue(x,θ), sim)
        u = r+c
    end
    return u::T
end

# -----------------------------------------
# deprecate
# -----------------------------------------

# @deprecate number_of_model_parms(x::DrillReward) _nparm(x)
# @deprecate coef_range_revenue(x)       idx_revenue(x)
# @deprecate coef_range_drillingcost(x)  idx_cost(x)
# @deprecate coef_range_extensioncost(x) idx_extend(x)
# @deprecate flowdθ!(args...) dflow!(args...)
# @deprecate theta_revenue(x::DrillReward, theta) vw_revenue(x, theta)
# @deprecate theta_cost(   x::DrillReward, theta) vw_cost(   x, theta)
# @deprecate theta_extend( x::DrillReward, theta) vw_extend( x, theta)
