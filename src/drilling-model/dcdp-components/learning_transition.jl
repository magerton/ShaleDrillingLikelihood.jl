
# ------------------------------ matrix updates -------------------------

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
            @. P[:,j] = _dcond_probdρ(yj,y,Δ,ρ) * dρdθρ
        elseif j == n
            @. P[:,j] = -_dcond_probdρ(yj,y,-Δ,ρ) * dρdθρ
        else
            @. P[:,j] = ( _dcond_probdρ(yj,y,Δ,ρ) - _dcond_probdρ(yj,y,-Δ,ρ) ) * dρdθρ
        end
    end
    return P
end


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
            @. P[ :,j] = normcdf(cond_z(yj, y, Δ, ρ))
        elseif j == n
            @. P[ :,j] = normccdf(cond_z(yj, y, -Δ, ρ))
        else
            @. P[ :,j] = ( normcdf(cond_z(yj, y, Δ, ρ)) - normcdf(cond_z(yj, y, -Δ, ρ) ))
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
