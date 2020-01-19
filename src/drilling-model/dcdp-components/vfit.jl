const VFTOL = 1e-10


"""
    vfit!(EV0, t, ddm)

Given `ubV(t) ≡ u + βV(x')`, update `EV0 ← E[max u + β V(x')]`
"""
function vfit!(EV0, t::DCDPTmpVars, ddm::DynamicDrillModel)
    if anticipate_t1ev(ddm)
        logsumexp_and_softmax!(lse(t), q(t), tmp(t), ubV(t))
    else
        maximum!(add_1_dim(lse(t)), ubV(t))
    end
    A_mul_B_md!(EV0, ztransition(ddm), lse(t), 1)
end

# preserves ubV & updates derivatives
function vfit!(EV0, dEV0, t::DCDPTmpVars, ddm::DynamicDrillModel)
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


function solve_inf_vfit!(EV0, t::DCDPTmpVars, ddm::DynamicDrillModel; maxit=60, vftol=VFTOL, kwargs...)

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

function div_and_update_direct!(EV0, t, ddm)
    J = size(EV0, 2)
    ΔEV = lse(t)
    q0 = ubV(t)

    for j in OneTo(J)
        q0j  = uview(q0, :, j, 1)
        ΔEVj = uview(ΔEV, :, j)

        update_IminusTVp!(t, ddm, q0j)
        fact = lu(IminusTEVp(t))
        ldiv!(fact, ΔEVj)  # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= ΔEV # update V
end

function div_and_update_indirect!(EV0, t, ddm, vftol; kwargs...)

    J = size(EV0,2)
    ΔEV = lse(t)
    q0 = ubV(t)
    ΔEVjnew = tmpEVj(t)

    for j in OneTo(J)
        q0j  = uview(q0, :, j, 1)
        ΔEVj = uview(ΔEV, :, j)
        if norm(ΔEVj, Inf) > vftol*1e-2  # b/c if ΔEVj ≈ 0, then we blow up
            fill!(ΔEVjnew, 0)
            update_IminusTVp!(t, ddm, q0j)
            # bicgstabl! results in NaNs
            gmres!(ΔEVjnew, IminusTEVp(t), ΔEVj; initially_zero=true, kwargs...)
            copyto!(ΔEVj, ΔEVjnew)
            # all(isfinite.(ΔEVjnew)) || throw(error("ΔEVjnew not finite. j=$j, ΔEVj = $ΔEVj, ΔEVjnew = $ΔEVjnew"))
        end
    end
    EV0 .-= ΔEV # update V
end


function pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, ddm::DynamicDrillModel; vftol=VFTOL, kwargs...)

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
        copyto!(EV0, tmp(t))
        return bnds
    end
    # Vtmp = [I - T'(V)] \ [V - T(V)]
    # V .= -Vtmp
    # div_and_update_direct!(EV0, t, ddm)
    div_and_update_indirect!(EV0, t, ddm, vftol; tol=1e-13)
    return extrema(ΔEV) .* -beta_1minusbeta(ddm) # get norm
end


# --------------------------- pfit until convergence ----------------------------


function solve_inf_pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, ddm::DynamicDrillModel; maxit=40, vftol=VFTOL, kwargs...)
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
function gradinf!(dEV0::AbstractArray3, t::DCDPTmpVars, ddm::DynamicDrillModel)

    nz, nψ, nk, nd = size(dubVperm(t))
    nd >= 2 || throw(error("Need dubV with at least 2+ action possibilities"))
    nψ >= nk  || throw(error("Need nψ >= length(theta)"))

    sumdubV   = view(dubVperm(t), :,:,:,1)
    ΠsumdubV  = view(dubVperm(t), :,:,:,2)

    ΠsumdubVj = view(lse(t), :, 1:nk) # Array{T}(nz,nθ)
    dev0tmpj  = view(tmp(t), :, 1:nk) # Array{T}(nz,nθ)

    if anticipate_t1ev(ddm)
        softmax3!(ubV(t), lse(t), tmp(t), ubV(t))
        # softmax3!(q, lse, tmp)
    else
        findmax!(add_1_dim(lse(t)), add_1_dim(tmp_cart(t)), ubV(t))
        fill!(ubV(t), 0)
        @inbounds for i in tmp_cart(t)
            setindex!(ubV(t), 1, i)
        end
    end

    # for dubV/dθt
    sumprod!(sumdubV, dubVperm(t), ubV(t))
    A_mul_B_md!(ΠsumdubV, ztransition(ddm), sumdubV, 1)
    # gradinf_inner_direct!(dEV0, dev0tmpj, ΠsumdubVj, ΠsumdubV, t, ddm)
    gradinf_inner_indirect!(dEV0, sumdubV, ΠsumdubV, t, ddm; gradinfmaxit=2)
end

function gradinf_inner_direct!(dEV0, dev0tmpj, ΠsumdubVj, ΠsumdubV, t, ddm)
    nψ = size(dEV0, 2)
    for j in OneTo(nψ)
        qj = view(ubV(t), :, j, 1)
        update_IminusTVp!(t, ddm, qj)

        fact = lu(IminusTEVp(t))

        # Note: cannot do this with @view(dEV0[:,j,:])
        ΠsumdubVj .= view(ΠsumdubV, :, j, :)
        ldiv!(dev0tmpj, fact, ΠsumdubVj) # ΠsumdubV[:,j,:])
        dEV0[:,j,:] .= dev0tmpj
    end
end


function gradinf_inner_indirect!(dEV0, sumdubV, ΠsumdubV, t, ddm; gradinfmaxit=1)
    nz, nψ, nθt = size(dEV0)

    # tmpvars
    βqdEV0 = sumdubV
    q0 = view(ubV(t), :, :, 1)

    # get some good starting values for iterative solver
    dEV0 .= ΠsumdubV
    for it = 1:gradinfmaxit
        βqdEV0 .= discount(ddm) .* q0 .* dEV0
        A_mul_B_md!(dEV0, ztransition(ddm), βqdEV0, 1)
        dEV0 .+= ΠsumdubV
    end

    # solve linear system for infinite horizon gradient
    for j in OneTo(nψ)
        qj = view(ubV(t), :, j, 1)
        update_IminusTVp!(t, ddm, qj)
        for k = OneTo(nθt)
            dEV0jk = uview(dEV0, :, j, k)
            ΠsumdubVjk = uview(ΠsumdubV, :, j, k)
            if any(dEV0jk != 0)
                gmres!(dEV0jk, IminusTEVp(t), ΠsumdubVjk; initially_zero=false)
            end
        end
    end
end

# # --------------------------- double vfit/pfit loop -----------------------

function solve_inf_vfit_pfit!(EV0::AbstractMatrix, t::DCDPTmpVars, prim::DynamicDrillModel; vftol=VFTOL, maxit0=45, maxit1=20, kwargs...)
    solve_inf_vfit!(EV0, t, prim; maxit=maxit0, vftol=vftol)

    # try-catch loop in case we have insane parameters that have Pr(no action) = 0, producing a singular IminusTEVp matrix.
    converged, iter, bnds = try
        solve_inf_pfit!(EV0, t, prim; maxit=maxit1, vftol=vftol)
    catch
        @warn "pfit threw error. trying vfit."
        converged, iter, bnds = solve_inf_vfit!(EV0, t, prim; maxit=5000, vftol=vftol)
    end
    return converged, iter, bnds
    # return solve_inf_pfit!(EV0, t, prim; maxit=maxit1, vftol=vftol)
end
