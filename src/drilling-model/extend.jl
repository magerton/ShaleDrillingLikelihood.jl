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
@inline function flow!(grad, ::ExtensionCost_Zero, d, obs, θ, sim, dograd::Bool)
    return azero(θ)
end

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
_nparm(::ExtensionCost_Constant) = 1
@inline function flow!(grad, ::ExtensionCost_Constant, d, obs, θ, sim, dograd::Bool)
    sgn = _sgnext(d, obs)
    if dograd
        grad[1] = _sgnext(d, obs)
    end
    return sgn ? θ[1] : azero(θ)
end
