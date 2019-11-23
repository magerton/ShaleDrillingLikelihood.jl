# --------------------------- basic VFIT ----------------------------

# simple Vfit
function vfit!(EV0, t::DCDPTmpVars, ddm::DynamicDrillingModel)
    if anticipate_t1ev(ddm)
        logsumexp_and_softmax!(lse(t), q(t), tmp(t), ubV(t))
    else
        maximum!(add_1_dim(lse(t)), ubV(t))
    end
    A_mul_B_md!(EV0, ztransition(ddm), lse(t), 1)
end

# preserves ubV & updates derivatives
function vfit!(EV0, dEV0, t::DCDPTmpVars, ddm::DynamicDrillingModel)
    if anticipate_t1ev(ddm)
        logsumexp_and_softmax!(lse(t), q(t), tmp(t), ubV(t))
    else
        findmax!( add_1_dim(lse(t)), add_1_dim(tmp_cart(t)), ubV(t) )
        fill!( q(t), 0)
        @inbounds @simd for i in tmp_cart(t)
            setindex!(q(t), 1, i)
        end
    end
    A_mul_B_md!(EV0, ztransition(ddm), lse(t), 1)

    sumdubV = view(dubVperm(t), :,:,:,1)
    sumprod!(sumdubV, dubVperm(t), q(t))
    A_mul_B_md!(dEV0, ztransition(ddm), sumdubV, 1)
end

# --------------------------- VFIT until conv ----------------------------


function solve_inf_vfit!(EV0, t::DCDPTmpVars, ddm::DynamicDrillingModel; maxit::Integer=60, vftol::Real=1e-11)

    iter = zero(maxit)
    while true
        vfit!(tmp(t), t, ddm)
        bnds = extrema( tmp(t) .- EV0 ) .* beta_1minusbeta(ddm)
        ubV(t)[:,:,1] .= discount(ddm) .* (EV0 .= tmp(t))
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
    end

end

# --------------------------- basic pfit ----------------------------

function pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, ddm::DynamicDrillingModel; vftol::Real=1e-11)

    ΔEV = lse(t)
    q0 = ubV(t)

    if anticipate_t1ev(ddm)
        logsumexp_and_softmax!(ΔEV, q0, tmp(t), ubV(t), 1)
    else
        findmax!(add_1_dim(ΔEV), add_1_dim(tmp_cart(t)), ubV(t))
        q0[:,:,1] .= last.(getfield.(tmp_cart(t), :I)) .== 1         # update q0 as Pr(d=0|x)
    end
    A_mul_B_md!(tmp(t), ztransition(ddm), ΔEV, 1)

    # compute difference & check bnds
    bnds = extrema(ΔEV .= EV0 .- tmp(t)) .* -beta_1minusbeta(ddm)
    if all(abs.(bnds) .< vftol)
        EV0 .= tmp(t)
        return bnds
    end

    # full PFit
    for j in OneTo(size(EV0, 2))
        q0j  = view(q0, :, j, 1)
        ΔEVj = view(ΔEV, :, j)

        update_IminusTVp!(t, ddm, q0j)
        fact = lu(IminusTEVp(t))
        ldiv!(fact, ΔEVj)                          # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= ΔEV                                  # update V
    return extrema(ΔEV) .* -beta_1minusbeta(ddm) # get norm
end

# --------------------------- pfit until convergence ----------------------------

function solve_inf_pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, ddm::DynamicDrillingModel; maxit::Integer=30, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        bnds = pfit!(EV0, t, ddm; vftol=vftol)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
        ubVfull(t)[:,:,1] .= discount(ddm) .* EV0
    end
end

# --------------------------- inf horizon gradient ----------------------------


# note -- this destroys ubV
function gradinf!(dEV0::AbstractArray3, t::DCDPTmpVars, ddm::DynamicDrillingModel)

    nz, nψ, nk, nd = size(dubVperm(t))
    nd >= 2 || throw(error("Need dubV with at least 2+ action possibilities"))
    nψ >= nk  || throw(error("Need nψ >= length(theta)"))

    sumdubV   = view(dubVperm(t), :,:,:,1)
    ΠsumdubV  = view(dubVperm(t), :,:,:,2)

    ΠsumdubVj = view(lse(t), :, 1:nk) # Array{T}(nz,nθ)
    dev0tmpj  = view(tmp(t), :, 1:nk) # Array{T}(nz,nθ)

    qq = ubV(t)

    if anticipate_t1ev(ddm)
        softmax3!(qq, lse(t), tmp(t), qq, 1)
        # softmax3!(q, lse, tmp)
    else
        findmax!(add_1_dim(lse(t)), add_1_dim(tmp_cart(t)), qq)
        fill!(qq, 0)
        @inbounds for i in tmp_cart(t)
            qq[i] = 1
        end
    end

    # for dubV/dθt
    sumprod!(sumdubV, dubVperm(t), qq)
    A_mul_B_md!(ΠsumdubV, ztransition(ddm), sumdubV, 1)

    for j in OneTo(nψ)
        qj = view(qq, :, j, 1)
        update_IminusTVp!(t, ddm, qj)
        fact = lu(IminusTEVp(t))

        # Note: cannot do this with @view(dEV0[:,j,:])
        ΠsumdubVj .= view(ΠsumdubV, :, j, :)
        ldiv!(dev0tmpj, fact, ΠsumdubVj) # ΠsumdubV[:,j,:])
        dEV0[:,j,:] .= dev0tmpj
    end
end

# function gradinf!(dEV0::AbstractArray3{T}, t::dcdp_tmpvars, prim::dcdp_primitives) where {T}
#     zeros2 = Array{T,2}(undef,0,0)
#     gradinf!(dEV0, zeros2, t, prim, false)
# end
#
# # --------------------------- double vfit/pfit loop -----------------------
#
function solve_inf_vfit_pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, prim::DynamicDrillingModel; vftol::Real=1e-9, maxit0::Integer=40, maxit1::Integer=20)
    solve_inf_vfit!(EV0, t, prim; maxit=maxit0, vftol=vftol)

    # try-catch loop in case we have insane parameters that have Pr(no action) = 0, producing a singular IminusTEVp matrix.
    converged, iter, bnds = # try
        solve_inf_pfit!(EV0, t, prim; maxit=maxit1, vftol=vftol)
    # catch
    #     @warn "pfit threw error. trying vfit."
    #     converged, iter, bnds = solve_inf_vfit!(EV0, t, prim; maxit=5000, vftol=vftol)
    # end
    return converged, iter, bnds
end

# --------------------------- helper function  ----------------------------


function sumprod!(red::AbstractArray3, big::AbstractArray4, small::AbstractArray3)
     nz, nψ, nk, nd = size(big)
    (nz, nψ,     nd) == size(small) || throw(DimensionMismatch())
    (nz, nψ, nk,   ) == size(red)   || throw(DimensionMismatch())

    # second set w/ plus equals
    @inbounds for d in 1:nd, k in 1:nk
        if d == 1
            @views red[:,:,k] .=  small[:,:,d] .* big[:,:,k,d]
        else
            @views red[:,:,k] .+= small[:,:,d] .* big[:,:,k,d]
        end
    end
end


function sumprod!(red::AbstractMatrix, big::AbstractArray3, small::AbstractArray3)
     nz, nψ, nd  = size(big)
    (nz, nψ, nd) == size(small) || throw(DimensionMismatch())
    (nz, nψ)     == size(red) || throw(DimensionMismatch())

    fill!(red, 0)
    @inbounds for d in 1:nd
        @views red .+= small[:,:,d] .* big[:,:,d]
    end
end
