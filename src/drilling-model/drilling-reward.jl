"""
`DrillReward` defines the reward function for basic drilling model.

It has 4 pieces. The corresponding parameter is
`theta_drill = [theta_DRILLcost, theta_EXTEND, theta_SCRAP, theta_REVENUE]`
"""
struct DrillReward{R<:AbstractDrillingRevenue,C<:AbstractDrillingCost,E<:AbstractExtensionCost,S<:AbstractScrapValue} <: AbstractStaticPayoff
    revenue::R
    drill::C
    extend::E
    scrap::S
end

DrillReward(r,d,e) = DrillReward(r,d,e,ScrapValue_Zero())

# access components
revenue(x::DrillReward) = x.revenue
drill(  x::DrillReward) = x.drill
extend( x::DrillReward) = x.extend
scrap(  x::DrillReward) = x.scrap
cost(   x::DrillReward) = drill(x)

# -----------------------------------------
# lengths
# -----------------------------------------

@inline _nparms(    x::DrillReward) = (_nparm(cost(x)), _nparm(extend(x)), _nparm(scrap(x)), _nparm(revenue(x)))
@inline _nparm(     x::DrillReward) = sum(_nparms(x))
@inline _nparm_cost_ext_scrap(x::DrillReward) = _nparm(cost(x)) + _nparm(extend(x)) + _nparm(scrap(x))
@inline _nparm_cost_ext(x::DrillReward) = _nparm(cost(x)) + _nparm(extend(x))

# `theta_drill = [theta_cost, theta_extend, theta_scrap, theta_revenue]`
idx_cost(   x::DrillReward) = OneTo(_nparm(cost(x)))
idx_extend( x::DrillReward) = OneTo(_nparm(extend(x)))  .+  _nparm(cost(x))
idx_scrap(  x::DrillReward) = OneTo(_nparm(scrap(x)))   .+ _nparm_cost_ext(x)
idx_revenue(x::DrillReward) = OneTo(_nparm(revenue(x))) .+ _nparm_cost_ext_scrap(x)
idx_ρ(      x::DrillReward) = _nparm(x) # idx_ρ(revenue(x), idx_revenue(x)

idx_drill_ρ(x::DrillReward) = idx_ρ(x)

# idx_g, idx_ψ, idx_t, idx_D are specific to each `revenue` function
idx_drill_g(x::DrillReward) = _nparm_cost_ext_scrap(x) + idx_g(revenue(x))
idx_drill_ψ(x::DrillReward) = _nparm_cost_ext_scrap(x) + idx_ψ(revenue(x))
idx_drill_t(x::DrillReward) = _nparm_cost_ext_scrap(x) .+ idx_t(revenue(x))
idx_drill_D(x::DrillReward) = _nparm_cost_ext_scrap(x) .+ idx_D(revenue(x))

# FIXME: define a revenue(d::DataDrill{<:AbstractDynamicDrillModel}) = revenue(reward(_model(d)))
idx_drill_g(d::DataDrill{<:AbstractDynamicDrillModel}) = idx_drill_g(reward(_model(d)))
idx_drill_ψ(d::DataDrill{<:AbstractDynamicDrillModel}) = idx_drill_ψ(reward(_model(d)))
idx_drill_t(d::DataDrill{<:AbstractDynamicDrillModel}) = idx_drill_t(reward(_model(d)))
idx_drill_D(d::DataDrill{<:AbstractDynamicDrillModel}) = idx_drill_D(reward(_model(d)))

vw_cost(   x::DrillReward, theta) = uview(theta, idx_cost(x))
vw_extend( x::DrillReward, theta) = uview(theta, idx_extend(x))
vw_scrap(  x::DrillReward, theta) = uview(theta, idx_scrap(x))
vw_revenue(x::DrillReward, theta) = uview(theta, idx_revenue(x))

# -----------------------------------------
# flow payoffs
# -----------------------------------------

function flow!(grad, x::DrillReward, d, obs, θ, sim, dograd)
    c = flow!(vw_cost(   x, grad), cost(   x), d, obs, vw_cost(   x, θ), sim, dograd)
    e = flow!(vw_extend( x, grad), extend( x), d, obs, vw_extend( x, θ), sim, dograd)
    s = flow!(vw_scrap(  x, grad), scrap(  x), d, obs, vw_scrap(  x, θ), sim, dograd)
    r = flow!(vw_revenue(x, grad), revenue(x), d, obs, vw_revenue(x, θ), sim, dograd)
    return c + e + s + r
end

"∂flow / ∂ψ for ψ¹ or ψ⁰"
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

function coefnames(x::DrillReward)
    c = coefnames(cost(   x))
    e = coefnames(extend( x))
    s = coefnames(scrap(  x))
    r = coefnames(revenue(x))
    return vcat(c, e, s, r)
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
