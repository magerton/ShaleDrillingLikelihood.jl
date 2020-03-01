export ExtensionCost_Constant,
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
        grad[1] = sgn
    end
    return sgn ? extensioncost(x,θ) : azero(θ)
end
coefnames(::ExtensionCost_Constant) = ["\\alpha_{ext}",]


# -------------------------------------------
# Scrap value
# -------------------------------------------

export ScrapValue_Zero, ScrapValue_Constant, ScrapValue_Price,
    ScrapValue_Constant_Discount

@inline dflow!(::AbstractScrapValue, args...) = nothing
@inline flowdψ(::AbstractScrapValue, args...) = 0.0

"No scrap value"
struct ScrapValue_Zero <: AbstractScrapValue end
_nparm(::ScrapValue_Zero) = 0
@inline function flow!(grad, x::ScrapValue_Zero, d, obs, θ, sim, dograd::Bool)
    return azero(θ)
end
coefnames(::ScrapValue_Zero) = Vector{String}(undef,0)



"Constant scrap value"
struct ScrapValue_Constant <: AbstractScrapValue end
_nparm(::ScrapValue_Constant) = 1
@inline function flow!(grad, x::ScrapValue_Constant, d, obs, θ, sim, dograd::Bool)
    e = expires_today(d, obs)
    if dograd
        grad[1] = e
    end
    return e ? θ[1] : azero(θ)
end
coefnames(::ScrapValue_Constant) = ["\\alpha_{scrap}",]


"Constant scrap value"
struct ScrapValue_Constant_Discount <: AbstractScrapValue end
_nparm(::ScrapValue_Constant_Discount) = 2
@inline function flow!(grad, x::ScrapValue_Constant_Discount, d, obs, θ, sim, dograd::Bool)
    e = expires_today(d, obs)
    if dograd
        grad[1] = e
        grad[2] = 0
    end
    return e ? θ[1] : azero(θ)
end
coefnames(::ScrapValue_Constant_Discount) = ["\\alpha_{scrap}", "\\alpha_{discount}"]


const DrillReward_Scrap_Const_Disc = DrillReward{R,C,E,ScrapValue_Constant_Discount} where {R,C,E}



"Scrap value with price"
struct ScrapValue_Price <: AbstractScrapValue end
_nparm(::ScrapValue_Price) = 2
@inline function flow!(grad, x::ScrapValue_Price, d, obs, θ, sim, dograd::Bool)
    e = expires_today(d, obs)
    p = exp(logprice(obs))
    if dograd
        grad[1] = e
        grad[2] = e*p
    end
    return e ? θ[1] + θ[2]*p : azero(θ)
end
coefnames(::ScrapValue_Price) = ["\\alpha_{scrap,0}", "\\alpha_{scrap,p}"]
