
# ------------------------------ matrix updates -------------------------

# recall, cond_z(yj,y,Δ,ρ) = z(ψ2|ψ1)
# cond_z(x2::Number, x1::Number, Δ::Number, ρ::Number) = (x2 - ρ*x1 + Δ)/sqrt(1-ρ^2)

"""
update `tmp.Πψtmp` as d(β*Πψ)/dθρ where Πψ is the tauchen approx of the 
transition matrix for dF(ψ⁰|ψ¹).

Elements are normpdf(zscore) * ∂zscore/∂y * ∂y/∂θ
"""
function _βΠψdθρ!(tmp, ddm, θ)
    P = Πψtmp(tmp)
    y = psispace(ddm)
    β = discount(ddm)
    n = checksquare(P)
    n == length(y) || throw(DimensionMismatch("size _Πψtmp = $n versus size(y) = $(size(y))"))

    rwrd = reward(ddm)
    σ = theta_ρ(rwrd,θ)
    ρ = _ρ(revenue(rwrd), σ)
    dρdθρ = _dρdθρ(σ)

    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[:,j] = β * _dcond_probdρ(yj,y,Δ,ρ) * dρdθρ
        elseif j == n
            @. P[:,j] = -β * _dcond_probdρ(yj,y,-Δ,ρ) * dρdθρ
        else
            @. P[:,j] = β * ( _dcond_probdρ(yj,y,Δ,ρ) - _dcond_probdρ(yj,y,-Δ,ρ) ) * dρdθρ
        end
    end
    return P
end


"""
update `tmp.Πψtmp` as β*Πψ where Πψ is the tauchen approx of the 
transition matrix for dF(ψ⁰|ψ¹). Is similar to `tauchen_1d!`
"""
function _βΠψ!(tmp, ddm, θ)
    P = Πψtmp(tmp)
    y = psispace(ddm)
    β = discount(ddm)
    n = checksquare(P)
    n == length(y) || throw(DimensionMismatch("size _Πψtmp = $n versus size(y) = $(size(y))"))

    rwrd = reward(ddm)
    σ = theta_ρ(rwrd,θ)
    ρ = _ρ(revenue(rwrd), σ)

    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * normcdf(cond_z(yj, y, Δ, ρ))
        elseif j == n
            @. P[ :,j] = β * normccdf(cond_z(yj, y, -Δ, ρ))
        else
            @. P[ :,j] = β * ( normcdf(cond_z(yj, y, Δ, ρ)) - normcdf(cond_z(yj, y, -Δ, ρ) ))
        end
    end
    return P
end

# ------------------------------ derivative check -------------------------

function check_dΠψ(ddm, θ)

    ψspace = psispace(ddm)

    rwrd = reward(ddm)
    θρ = theta_ρ(rwrd,θ)
    ρ = _ρ(revenue(rwrd), θρ)


    Δ = 0.5 * step(ψspace)

    for yj in ψspace
        for y in ψspace
            fdσ = Calculus.derivative((sig) -> normcdf(cond_z(yj, y, Δ, _ρ(sig))), θρ)
            dσ  = _dρdθρ(θρ) * _dcond_probdρ(yj, y, Δ, ρ)
            abs(fdσ - dσ ) < 1e-7 || throw(error("bad σ grad at σ = $σ, ψ2 = $yj, ψ1 = $y"))
        end
    end
    return true
end
