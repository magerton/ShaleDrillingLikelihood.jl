export flow, dflow!, flowdσ, flowdψ

# -----------------------------------------
# components of stuff
# -----------------------------------------

const DrillModel = Union{DynamicDrillingModel,DrillReward}

# access components
revenue(x::DrillReward) = x.revenue
drill(  x::DrillReward) = x.drill
extend( x::DrillReward) = x.extend
cost(   x::DrillReward) = drill(x)

@deprecate revenue(x::ObservationDrill) revenue(_model(x))
@deprecate drill(  x::ObservationDrill) drill(  _model(x))
@deprecate extend( x::ObservationDrill) extend( _model(x))
@deprecate extensioncost(x::DrillModel) extend(x)
@deprecate drillingcost( x::DrillModel) drill(x)

# -----------------------------------------
# lengths
# -----------------------------------------

@inline _nparms(x::DrillReward) = (_nparm(revenue(x)), _nparm(drill(x)), _nparm(extend(x)),)
@inline _nparm( x::DrillReward) = sum(_nparms(x))

@inline idx_revenue(x::DrillReward) = OneTo(_nparm(revenue(x)))
@inline idx_cost(   x::DrillReward) = OneTo(_nparm(drill(x)))  .+  _nparm(revenue(x))
@inline idx_extend( x::DrillReward) = OneTo(_nparm(extend(x))) .+ (_nparm(revenue(x)) + _nparm(drill(x)))

@inline theta_drill_indexes(x::DrillReward) = (idx_revenue(x), idx_cost(x), idx_extend(x),)

@deprecate number_of_model_parms(x::DrillReward) _nparm(x)
@deprecate coef_range_revenue(x)       idx_revenue(x)
@deprecate coef_range_drillingcost(x)  idx_cost(x)
@deprecate coef_range_extensioncost(x) idx_extend(x)
@deprecate coef_ranges(x)              theta_drill_indexes(x)

@deprecate flowdθ!(args...) dflow!(args...)

@inline check_coef_length(x::DrillReward, θ) = (_nparm(x) == length(θ) || throw(DimensionMismatch()))

vw_revenue(x::DrillReward, theta) = view(theta, idx_revenue(x))
vw_cost(   x::DrillReward, theta) = view(theta, idx_cost(x))
vw_extend( x::DrillReward, theta) = view(theta, idx_extend(x))

@deprecate theta_revenue(x::DrillReward, theta) vw_revenue(x, theta)
@deprecate theta_cost(   x::DrillReward, theta) vw_cost(   x, theta)
@deprecate theta_extend( x::DrillReward, theta) vw_extend( x, theta)

# -----------------------------------------
# flow payoffs
# -----------------------------------------

@inline function flow(x::DrillReward, d, obs, θ, sim)
    T = eltype(θ)
    if d == 0
        u = flow(extend(x),  d, obs, vw_extend(x,θ), sim)
    else
        u = flow(cost(x),    d, obs, vw_cost(x,θ), sim) +
            flow(revenue(x), d, obs, vw_revenue(x,θ), sim)
    end
    return u::T
end

@inline function dflow!(x::DrillReward, grad, d, obs, θ, sim)
    if d == 0
        dflow!(extend( x), vw_extend( x, grad), d, obs, vw_extend( x, θ), sim)
    else
        dflow!(cost(   x), vw_cost(   x, grad), d, obs, vw_cost(   x, θ), sim)
        dflow!(revenue(x), vw_revenue(x, grad), d, obs, vw_revenue(x, θ), sim)
    end
    return nothing
end

@inline function flowdψ(x::DrillReward, d, obs, theta, sim)
    if d == 0
        return flowdψ(extend(x), d, obs, vw_extend(x,theta), sim)
    else
        c = flowdψ(cost(x),    d, obs, vw_cost(x,theta), sim)
        r = flowdψ(revenue(x), d, obs, vw_revenue(x,theta), sim)
        return r+c
    end
end
