ThetaRho() = 0.0
AlphaPsi() = STARTING_α_ψ
AlphaG() = STARTING_log_ogip
AlphaT() = STARTING_α_t
Alpha0() = -2.8


Theta(m, args...) = throw(error("Default Theta not assigned for $m"))
Theta(m::AbstractDynamicDrillModel, args...) = Theta(reward(m), args...)
function Theta(m::DrillReward, args...)
    c = Theta(cost(m), args...)
    e = Theta(extend(m), args...)
    r = Theta(revenue(m), args...)
    return vcat(c,e,r)
end


# Royalty
function Theta(m::RoyaltyModel; θρ=ThetaRho(), kwargs...)
    #            θρ  ψ        g    x_r      κ₁   κ₂
    return vcat(θρ, 1.0,    0.1, -0.3,    -0.6, 0.6)
end

# Production
function Theta(m::ProductionModel; αψ=AlphaPsi(), αg=AlphaG(), kwargs...)
    #           αψ, αg, γx   σ2η, σ2u  # NOTE: η is iwt, u is iw
    return vcat(αψ, αg, 0.2, 0.3, 0.4)
end

# Test Drill
function Theta(m::TestDrillModel; θρ=ThetaRho(), αψ=AlphaPsi(), kwargs...)
    return vcat(αψ, 2.0, -2.0, -0.75, θρ)
end


# revenue
# ---------
function Theta(m::DrillingRevenue{Constrained}; θρ=ThetaRho(), kwargs...)
    return vcat(Alpha0(), θρ)
end

function Theta(m::DrillingRevenue{Unconstrained, NoTrend};
    θρ=ThetaRho(), αψ=AlphaPsi(), αg=AlphaG(), α0=Alpha0(), kwargs...)
    return vcat(α0, αg, αψ, θρ)
end

function Theta(m::DrillingRevenue{Unconstrained, TimeTrend};
    θρ=ThetaRho(), αψ=AlphaPsi(), αg=AlphaG(), αT=AlphaT(), α0=Alpha0(), kwargs...)
    return vcat(α0, αg, αψ, αt, θρ)
end

# cost
# ---------
Theta(m::DrillingCost_constant, args...) = vcat(-5.5)

# extension
# ---------
Theta(m::ExtensionCost_Zero    , args...) = zeros(0)
Theta(m::ExtensionCost_Constant, args...) = vcat(-2.0)


CoefLinks(r) = (zeros(Int,0), zeros(Int,0))

CoefLinks(r::DrillingRevenue{Unconstrained, NoTrend}) =
    [(idx_produce_ψ, idx_drill_ψ,), (idx_produce_g, idx_drill_g)]

CoefLinks(r::DrillingRevenue{Unconstrained, TimeTrend}) =
    [
        (idx_produce_ψ, idx_drill_ψ,),
        (idx_produce_g, idx_drill_g),
        (idx_produce_g, idx_drill_g),
    ]

CoefLinks(r::DrillReward) = CoefLinks(revenue(r))
CoefLinks(m::DynamicDrillModel) = CoefLinks(reward(m))
CoefLinks(d::DataDrill) = CoefLinks(_model(d))
