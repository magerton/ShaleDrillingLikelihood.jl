abstract type AbstractEstimationTempvars end

# ------------------------------------------------------
# Production
# ------------------------------------------------------

struct ProductionTempVars <: AbstractEstimationTempvars
    v::Vector{Float64}
    Xpv::Vector{Float64}
    Xp1::Vector{Float64}
end

_v(ptv::ProductionTempVars) = ptv.v
_Xpv(ptv::ProductionTempVars) = ptv.Xpv




update_Xpv!(tmp_pdxn) = mul!(Xpv(tmp_pdxn), X(tmp_pdxn), v(tmp_pdxn))

function update_v!(v, y, X, β)
    v .= y
    BLAS.gemv!('T', -1.0, X, β, 1.0, v)
end


function update_v!(ptv::ProductionTempVars, model, theta, data)
    _v(ptv) .= _ypdxn(data) .- _Xpdxn(data)*theta_pdxn_β(model, theta)
end

# ------------------------------------------------------
# Royalty
# ------------------------------------------------------


struct RoyaltyTempVars <: AbstractEstimationTempvars
    xbeta::Vector{Float64}
end

_xbeta(rtv::RoyaltyTempVars) = rtv.xbeta

function update!(rtv::RoyaltyTempVars, model, theta, data)
    mul!(_xbeta(RoyaltyTempVars), _Xroy(data), theta_roy_β(model,theta))
end
