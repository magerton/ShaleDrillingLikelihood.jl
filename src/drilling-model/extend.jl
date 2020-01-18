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
extensioncost(::ExtensionCost_Zero, θ) = azero(θ)
@inline function flow!(grad, x::ExtensionCost_Zero, d, obs, θ, sim, dograd::Bool)
    return extensioncost(x,θ)
end
coefnames(::ExtensionCost_Zero) = Vector{String}(undef,0)

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
_nparm(::ExtensionCost_Constant) = 1
extensioncost(::ExtensionCost_Constant, θ) = θ[1]
@inline function flow!(grad, x::ExtensionCost_Constant, d, obs, θ, sim, dograd::Bool)
    sgn = _sgnext(d, obs)
    if dograd
        grad[1] = _sgnext(d, obs)
    end
    return sgn ? extensioncost(x,θ) : azero(θ)
end
coefnames(::ExtensionCost_Constant) = ["\\alpha_{ext}",]
