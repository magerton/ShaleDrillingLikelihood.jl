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
length(::ExtensionCost_Zero) = 0
@inline flow(  ::ExtensionCost_Zero, d, obs, θ, sim) = azero(θ)
@inline dflow!(::ExtensionCost_Zero, d, obs, θ, sim) = nothing

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
length(::ExtensionCost_Constant) = 1
@inline flow(  ::ExtensionCost_Constant, d, obs, θ, sim) = _sgnext(d, obs) ? θ[1] : azero(θ)
@inline function dflow!(::ExtensionCost_Constant, grad, d, obs, θ, sim)
    if _sgnext(d, obs)
        grad[1] += 1
    end
    return nothing
end

# "Extension cost depends on ψ"
# struct ExtensionCost_ψ <: AbstractExtensionCost end
# length(::ExtensionCost_ψ) = 2
# @inline flow(  ::ExtensionCost_ψ, d, obs, θ, sim) = θ[1] + θ[2] * _ψ(obs, sim)
# @inline flowdψ(::ExtensionCost_ψ, d, obs, θ, sim) = θ[2]
# @inline function dflow!(::ExtensionCost_ψ, grad, d, obs, θ, sim)
#     if _sgnext(d, obs)
#         grad[1] += 1
#         grad[2] += _ψ(obs, sim)
#     end
#     return nothing
# end
