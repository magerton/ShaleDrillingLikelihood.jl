export ThetaConstrained

ThetaRho() = 0.0
AlphaPsi() = STARTING_α_ψ
AlphaG() = STARTING_log_ogip
AlphaT() = STARTING_α_t
Alpha0() = -2.8
BetaPsi() = 0.11805850128182346

SigmaSq_eta() = 0.009368079566227127
SigmaSq_u() = 0.10145638253519064
Gamma0() = -14.951888468589365

Theta(m, args...) = throw(error("Default Theta not assigned for $m"))
Theta(m::AbstractDynamicDrillModel, args...; kwargs...) = Theta(reward(m), args...; kwargs...)
function Theta(m::DrillReward, args...; kwargs...)
    c = Theta(cost(m), args...; kwargs...)
    e = Theta(extend(m), args...; kwargs...)
    r = Theta(revenue(m), args...; kwargs...)
    return vcat(c,e,r)
end

function ThetaConstrained(m::DrillReward{<:DrillingRevenueUnconstrained}, theta)
    rwrd_c = ConstrainedProblem(m, theta)
    thet_c = [x for (i,x) in enumerate(theta) if i ∉ ShaleDrillingLikelihood.ConstrainedIdx(m)]
    return thet_c
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

# from datasets
# ---------

function Theta(d::DataRoyalty; θρ=ThetaRho(), αψ=AlphaPsi(), kwargs...)
    βψ = BetaPsi()
    if num_choices(d) == 6 && _num_x(d) == 4
        betas = [ 0.5966055028536498, 1.1822487035901503, -1.7017414563283797, 0.14048454874430671]
        kappas = [3.8373126232392716, 4.172804362649826, 5.0264099395765385, 5.925934015361181, 6.501397017005945]
        return vcat(θρ, βψ, betas, kappas)
    end
    throw(error("no starting values available"))
end

function Theta(d::DataProduce; αψ=AlphaPsi(), αg=AlphaG(), αt=AlphaT(),
        σ2η=SigmaSq_eta(), σ2u=SigmaSq_u(), γ0=Gamma0(), kwargs...)
    if _num_x(d) == 2
        return vcat(αψ, αg, γ0,     σ2η, σ2u)
    elseif _num_x(d) == 3
        return vcat(αψ, αg, γ0, αt, σ2η, σ2u)
    end
    throw(error("no starting values available"))
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

Theta(m::DrillingCost_constant, args...; kwargs...) = vcat(-5.5)

function Theta(m::DrillingCost_TimeFE, args...; kwargs...)
    c2plus = 1.407
    if ShaleDrillingLikelihood.start(m) == 2008 && ShaleDrillingLikelihood.stop(m) == 2012
        timefe = [-9.48651914568503, -6.198466654405615, -4.859543515359247, -4.391926980826075, -4.464122201335934,]
    else
        throw(error("no starting values for start $(start(m)) and stop $(stop(m))"))
    end
    return vcat(timefe, c2plus)
end


# extension
# ---------
Theta(m::ExtensionCost_Zero    , args...; kwargs...) = zeros(0)
Theta(m::ExtensionCost_Constant, args...; kwargs...) = vcat(-1.0)


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
