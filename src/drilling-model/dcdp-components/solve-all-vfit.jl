function solve_vf_terminal!(evs, ddm, dograd)
    ev = EV(evs)
    dev = dEV(evs)

    exp_trm = exploratory_terminal(statespace(ddm))

     ev[:, :, end    ] .= 0
     ev[:, :, exp_trm] .= 0

     if dograd
         dev[:, :, :, end    ] .= 0
         dev[:, :, :, exp_trm] .= 0
    end
    return nothing
end

# ---------------------------------------------

function solve_vf_infill!(evs, t::DCDPTmpVars, ddm::DynamicDrillingModel, θ, ichar, dograd; kwargs...)

    wp = statespace(ddm)

    for i in ind_inf(wp)
        idxd, idxs, horzn = dp1space(wp,i), collect(sprimes(wp,i)), _horizon(wp,i)

        tvw = view(t, idxd)
        @views EV0  = view( EV(evs), :, :,    i)
        @views dEV0 = view(dEV(evs), :, :, :, i)

        flow!(tvw, ddm, θ, i, ichar, dograd)
        ubV(tvw) .+= discount(ddm) .* view(EV(evs), :, :, idxs)

        if dograd
            fill!(dEV0, 0)
            dubVperm(tvw) .+= discount(ddm) .* view(dEV(evs), :, :, :, idxs)
        end

        if horzn == :Finite
            if dograd
                vfit!(EV0, dEV0, tvw, ddm)
            else
                vfit!(EV0, tvw, ddm)
            end

        elseif horzn == :Infinite
            converged, iter, bnds =  solve_inf_vfit_pfit!(EV0, tvw, ddm; kwargs...)
            converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds. θ = $θ"

            if dograd
                # TODO: only allows 0-payoff if no action
                ubV(tvw)[:,:,1] .= discount(ddm) .* EV0
                gradinf!(dEV0, tvw, ddm)   # note: destroys ubV & dubV
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
    return nothing
end


# ---------------------------------------------


function learningUpdate!(evs, t::DCDPTmpVars, ddm::DynamicDrillingModel, θ, ichar, dograd)

    wp = statespace(ddm)
    lrn2inf = inf_fm_lrn(wp)
    exp2lrn = exploratory_learning(wp)

    EV1   = view( EV(evs), :, :,    lrn2inf)
    EV2   = view( EV(evs), :, :,    exp2lrn)
    dEV1  = view(dEV(evs), :, :, :, lrn2inf)
    dEV2  = view(dEV(evs), :, :, :, exp2lrn)

    idx_σ = idx_ρ(reward(ddm))
    dEVσ  = view(dEV(evs), :, :, idx_σ, exp2lrn)

    _βΠψ!(t, ddm, θ)
    A_mul_B_md!(EV2, Πψtmp(t), EV1, 2)

    if dograd
        # dubVtilde/dθ = du/dθ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dθ[:,:,:,2:dmaxp1]
        A_mul_B_md!(dEV2, Πψtmp(t), dEV1, 2)

        # ∂EVtilde/∂σ[:,:,2:dmaxp1] = ∂u/∂σ[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
        _βΠψdθρ!(t, ddm, θ)
        A_mul_B_md!(dEVσ, Πψtmp(t), EV1, 2)
    end
    return nothing
end


# ---------------------------------------------


function solve_vf_explore!(evs, t::DCDPTmpVars, ddm::DynamicDrillingModel, θ, ichar, dograd; kwargs...)

    wp = statespace(ddm)
    dmaxp1 = _dmax(wp)+1
    exp2lrn = exploratory_learning(wp)

    # Views of ubV so we can efficiently access them
    tvw = view(t, OneTo(dmaxp1))

     ubV0 = view(ubV(tvw),      :, :,    1)
     ubV1 = view(ubV(tvw),      :, :,    2:dmaxp1)
    dubV0 = view(dubVperm(tvw), :, :, :, 1)
    dubV1 = view(dubVperm(tvw), :, :, :, 2:dmaxp1)
    βEV1  = view( EV(evs),      :, :,    exp2lrn)
    βdEV1 = view(dEV(evs),      :, :, :, exp2lrn)

    # do VFI
    for i in ind_exp(wp)
        ip0 = sprime(wp,i,0)    # s_{t+1} if d_t = 0
        horzn = _horizon(wp,i)

         EV0_ip = view( EV(evs), :, :,    ip0)
        dEV0_ip = view(dEV(evs), :, :, :, ip0)

        # compute u + βEV(d) ∀ d ∈ actionspace(wp,i)
        flow!(tvw, ddm, θ, i, ichar, dograd)
        ubV0 .+= discount(ddm) .* EV0_ip
        ubV1 .+= βEV1 # β already baked in

        if dograd
            dubV0 .+= discount(ddm) .* dEV0_ip
            dubV1 .+= βdEV1  # β already baked in
        end

        if horzn == :Finite
            EV0_i = view(EV(evs), :, :, i)
            if dograd
                dEV0_i = view(dEV(evs), :, :, :, i)
                @views vfit!(EV0_i, dEV0_i, tvw, ddm)
            else
                @views vfit!(EV0_i, tvw, ddm)
            end

        elseif horzn == :Infinite
            converged, iter, bnds =  solve_inf_vfit_pfit!(EV0_ip, tvw, ddm; kwargs...)
            converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds. θt = $θt, σ = $σ"

            if dograd
                # TODO: only allows 0-payoff if no action
                ubV(tvw)[:,:,1] .= discount(ddm) .* EV0_ip
                gradinf!(dEV0_ip, tvw, ddm)
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
    return nothing
end

# ---------------------------------------------

function solve_vf_all!(evs, t, ddm, θ, ichar, dograd; kwargs...)
    solve_vf_terminal!(evs,    ddm,           dograd; kwargs...)
    solve_vf_infill!(  evs, t, ddm, θ, ichar, dograd; kwargs...)
    learningUpdate!(   evs, t, ddm, θ, ichar, dograd; kwargs...)
    solve_vf_explore!( evs, t, ddm, θ, ichar, dograd; kwargs...)
    return nothing
end

function solve_vf_all_timing!(evs, args...)
    fill!(evs, 0)
    solve_vf_all!(evs, args...)
    return nothing
end
