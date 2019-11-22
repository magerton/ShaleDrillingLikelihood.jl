export AbstractExtensionCost,
    ExtensionCost_Constant,
    ExtensionCost_Zero

# -------------------------------------------
# Extension
# -------------------------------------------

@inline dflow!(::AbstractExtensionCost, args...) = nothing
@inline flowdψ(::AbstractExtensionCost, args...) = 0.0

"No extension cost"
struct ExtensionCost_Zero <: AbstractExtensionCost end
_nparm(::ExtensionCost_Zero) = 0
@inline flow(  ::ExtensionCost_Zero, d, obs, θ, sim) = azero(θ)
@inline dflow!(::ExtensionCost_Zero, d, obs, θ, sim) = nothing

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
_nparm(::ExtensionCost_Constant) = 1
@inline flow(  ::ExtensionCost_Constant, d, obs, θ, sim) = _sgnext(d, obs) ? θ[1] : azero(θ)
@inline function dflow!(::ExtensionCost_Constant, grad, d, obs, θ, sim)
    grad[1] = _sgnext(d, obs)
    return nothing
end
