export flow, dflow!, flowdσ, flowdψ,

# -----------------------------------------
# components of stuff
# -----------------------------------------

# access components
revenue(x::DrillModel) = x.revenue
drill(  x::DrillModel) = x.drill
extend( x::DrillModel) = x.extend
cost(   x::DrillModel) = drill(x)

revenue(x::ObservationDrill) = revenue(_model(x))
drill(  x::ObservationDrill) = drill(  _model(x))
extend( x::ObservationDrill) = extend( _model(x))


@deprecate extensioncost(x::DrillModel) extend(x)
@deprecate drillingcost( x::DrillModel) drill(x)

# -----------------------------------------
# lengths
# -----------------------------------------

@inline lengths(x::DrillModel) = (length(revenue(x)), length(drill(x)), length(extend(x)),)
@inline length( x::DrillModel) = sum(lengths(x))
@inline _nparm(x::DrillModel) = length(x)
@deprecate number_of_model_parms(x::DrillModel) _nparm(x)

# -----------------------------------------
# lengths
# -----------------------------------------
# coeficient ranges
@inline idx_revenue(x::DrillModel) = OneTo(length(revenue(x)))
@inline idx_cost(   x::DrillModel) = OneTo(length(drill(x)))  .+  length(revenue(x))
@inline idx_extend( x::DrillModel) = OneTo(length(extend(x))) .+ (length(revenue(x)) + length(drill(x)))
@inline theta_drill_indexes(x::DrillModel) = (idx_revenue(x), idx_cost(x), idx_extend(x),)

@deprecate coef_range_revenue(x)       idx_revenue(x)
@deprecate coef_range_drillingcost(x)  idx_cost(x)
@deprecate coef_range_extensioncost(x) idx_extend(x)
@deprecate coef_ranges(x)              theta_drill_indexes(x)

@deprecate flowdθ!(args...) dflow!(args...)

@inline check_coef_length(x::DrillModel, θ) = (length(x) == length(θ) || throw(DimensionMismatch()))

vw_revenue(x::DrillModel, theta) = view(theta, idx_revenue(x))
vw_cost(   x::DrillModel, theta) = view(theta, idx_cost(x))
vw_extend( x::DrillModel, theta) = view(theta, idx_extend(x))

@deprecate theta_revenue(x::DrillModel, theta) vw_revenue(x, theta)
@deprecate theta_cost(   x::DrillModel, theta) vw_cost(   x, theta)
@deprecate theta_extend( x::DrillModel, theta) vw_extend( x, theta)

@inline function flow(x::DrillModel, d, obs, θ, sim)
    T = eltype(θ)
    if d == 0
        u = flow(extend(x),  d, obs, vw_extend(x,θ), sim)
    else
        u = flow(revenue(x), d, obs, vw_revenue(x,θ), sim) +
            flow(cost(x),    d, obs, vw_cost(x,θ), sim)
    end
    return u::T
end

@inline function dflow!(x::DrillModel, grad, d, obs, θ, sim)
    if d == 0
        dflow!(extend( x), vw_extend( x, grad), d, obs, vw_extend( x, θ), sim)
    else
        dflow!(cost(   x), vw_cost(   x, grad), d, obs, vw_cost(   x, θ), sim)
        dflow!(revenue(x), vw_revenue(x, grad), d, obs, vw_revenue(x, θ), sim)
    end
    return nothing
end

@inline function flowdψ(x::DrillModel, d, obs, theta, sim)
    if d == 0
        return flowdψ(extend(x), d, obs, vw_extend(x,theta), sim)
    else
        r = flowdψ(revenue(x), d, obs, vw_revenue(x,theta), sim)
        c = flowdψ(cost(x),   d, obs, vw_cost(x,theta), sim)
        return r+c
    end
end
