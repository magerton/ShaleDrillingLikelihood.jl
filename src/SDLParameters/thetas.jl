using ShaleDrillingLikelihood: vw_revenue, vw_cost, vw_extend,
    DrillingRevenueUnconstrained, DrillingRevenueConstrained,
    ConstrainedIdx, UnconstrainedFmConstrainedIdx

export updateThetaUnconstrained!, ThetaConstrained, ThetaRho

ThetaRho() = 0.726029
AlphaPsi() = 0x1.55b10dcc51d5fp-2
AlphaG() = 0x1.075c51cf96449p-1
AlphaT() = STARTING_α_t
Alpha0() = -2.8
BetaPsi() = 0.11805850128182346
AlphaTFE() = [ 0.20665908809883377, 0.1661000339389365, 0.14007105659535182,
    0.14457016546750073, 0.19677616668888495, 0.26422850311680984,
    0.5206302285265718, 0.635563439673686,]
Sigma_eta() = 0x1.8c7275fb0e3d2p-4
Sigma_u() = 0x1.2ca92ef5843d8p-2
Gamma0() = -0x1.c98f8c57dd39bp+3

@deprecate SigmaSq_eta() Sigma_eta()
@deprecate SigmaSq_u() Sigma_u()

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
    theta_rev = vw_revenue(m, theta)
    r = [x for (i,x) in enumerate(theta_rev) if i ∉ ConstrainedIdx(revenue(m))]
    c = vw_cost(m,theta)
    e = vw_extend(m,theta)
    return vcat(c,e,r)
end

function updateThetaUnconstrained!(m::DrillReward{<:DrillingRevenueConstrained}, thetau, thetac)
    rwrd_u = UnconstrainedProblem(m)
    _nparm(m) == length(thetac) ||
        throw(DimensionMismatch("_nparm(m) = $(_nparm(m)) != length(thetac) = $(length(thetac))"))
    _nparm(rwrd_u) == length(thetau) ||
        throw(DimensionMismatch("_nparm(rwrd_u) = $(_nparm(rwrd_u)) != length(thetau) = $(length(thetau))"))
    vw_cost(rwrd_u, thetau) .= vw_cost(m, thetac)
    vw_extend(rwrd_u, thetau) .= vw_extend(m, thetac)

    revidx = UnconstrainedFmConstrainedIdx(revenue(m))
    thetarev_u = vw_revenue(rwrd_u, thetau)
    thetarev_u[revidx] .= vw_revenue(m, thetac)
    return thetau
end

function updateThetaUnconstrained!(m::DrillReward{<:DrillingRevenueUnconstrained}, thetau, thetac)
    rwrd_c = ConstrainedProblem(m)
    return updateThetaUnconstrained!(rwrd_c, thetau, thetac)
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
         αTFE=AlphaTFE(), ση=Sigma_eta(), σu=Sigma_u(), γ0=Gamma0(), kwargs...)
    if _num_x(d) == 2
        return vcat(αψ, αg, γ0,     ση, σu)
    elseif _num_x(d) == 3
        return vcat(αψ, αg, αt, γ0, ση, σu)
    elseif _num_x(d) == 10
        return vcat(αψ, αg, αTFE, γ0, ση, σu)
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
    return vcat(α0, αg, αψ, αT, θρ)
end

function Theta(m::DrillingRevenue{Unconstrained, TimeFE};
    θρ=ThetaRho(), αψ=AlphaPsi(), αg=AlphaG(), αT=AlphaT(), αTFE=AlphaTFE(), α0=Alpha0(), kwargs...)
    _nparm(tech(m)) == length(αTFE) || throw(DimensionMismatch())
    return vcat(α0, αg, αψ, αTFE, θρ)
end


# cost
# ---------

Theta(m::DrillingCost_constant, args...; kwargs...) = vcat(-5.5)

Theta(m::DrillingCost_dgt1, args...; kwargs...) = vcat(-5.5, 1.0)

function Theta(m::DrillingCost_TimeFE, args...; kwargs...)
    c2plus = 1.407
    n = length(ShaleDrillingLikelihood.yearrange(m))
    if n == 5
        timefe = [-9.48651914568503, -6.198466654405615, -4.859543515359247, -4.391926980826075, -4.464122201335934,]
    else
        throw(error("no starting values for length(yearrange($m)) == 5"))
    end
    return vcat(timefe, c2plus)
end

function Theta(m::DrillingCost_TimeFE_rigrate, args...; kwargs...)
    c2plus = 1.407
    crig = -1.0
    n = length(ShaleDrillingLikelihood.yearrange(m))
    if n == 5
        timefe = [-9.48651914568503, -6.198466654405615, -4.859543515359247, -4.391926980826075, -4.464122201335934,]
    else
        throw(error("no starting values for length(yearrange($m)) == 5"))
    end
    return vcat(timefe, c2plus, crig)
end


function Theta(m::DrillingCost_TimeFE_costdiffs, args...; kwargs...)
    n = length(ShaleDrillingLikelihood.yearrange(m))
    if n == 5
        timefe = [-10.548923591552184, -7.034018996118814, -5.758999044446409, -5.539143772743451, -5.579481358288621,]
    else
        throw(error("no starting values for length(yearrange($m)) == 5"))
    end
    return vcat(timefe, 0.017248457193998108, -1.155685021020903, 1.6270414642692912,)
end

# extension
# ---------
Theta(m::ExtensionCost_Zero    , args...; kwargs...) = zeros(0)
Theta(m::ExtensionCost_Constant, args...; kwargs...) = vcat(-1.0)


CoefLinks(r) = (zeros(Int,0), zeros(Int,0))

CoefLinks(r::DrillReward{<:DrillingRevenue{Unconstrained, NoTrend}}) =
    [(idx_produce_ψ, idx_drill_ψ,), (idx_produce_g, idx_drill_g)]

CoefLinks(r::DrillReward{<:DrillingRevenue{Unconstrained, TimeTrend}}) =
    [
        (idx_produce_ψ, idx_drill_ψ,),
        (idx_produce_g, idx_drill_g),
        (idx_produce_t, idx_drill_t),
    ]

function CoefLinks(r::DrillReward{<:DrillingRevenue{Unconstrained, TimeFE}})
    idx_d_t = idx_drill_t(r)
    idx_p_t = idx_produce_t() .+ (idx_d_t .- first(idx_d_t))
    cfd = vcat(idx_drill_ψ(r), idx_drill_g(r), idx_d_t)
    cfp = vcat(idx_produce_ψ(), idx_produce_g(), idx_p_t)
    return (cfp, cfd,)
end

# CoefLinks(r::DrillReward) = CoefLinks(revenue(r))
CoefLinks(m::DynamicDrillModel) = CoefLinks(reward(m))
CoefLinks(d::DataDrill) = CoefLinks(_model(d))
