function solve_vf_terminal!(ddm::DDM_AbstractVF, dograd)
    evs = value_function(ddm)
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

function solve_vf_infill!(t::DCDPTmpVars, ddm::DDM_AbstractVF, θ, ichar, dograd; kwargs...)

    evs = value_function(ddm)
    wp = statespace(ddm)

    for i in ind_inf(wp)
        idxd, idxs, horzn = dp1space(wp,i), collect(sprimes(wp,i)), _horizon(wp,i)

        tvw = view(t, idxd)
        @views EV0  = view( EV(evs), :, :,    i)
        @views dEV0 = view(dEV(evs), :, :, :, i)

        update_static_payoffs!(tvw, ddm, θ, i, ichar, dograd)
        _ubv = ubV(tvw)
        _ubv .+= discount(ddm) .* view(EV(evs), :, :, idxs)

        if dograd
            fill!(dEV0, 0)
            _dubv = dubVperm(tvw)
            _dubv .+= discount(ddm) .* view(dEV(evs), :, :, :, idxs)
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
                _ubv[:,:,1] .= discount(ddm) .* EV0
                gradinf!(dEV0, tvw, ddm)   # note: destroys ubV & dubV
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
    return nothing
end

# ---------------------------------------------


function learningUpdate!(t::DCDPTmpVars, ddm::DDM_AbstractVF, θ, ichar, dograd)
    evs = value_function(ddm)
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


function solve_vf_explore!(t::DCDPTmpVars, ddm::DDM_AbstractVF, θ, ichar, dograd; kwargs...)

    evs = value_function(ddm)
    wp = statespace(ddm)
    dmaxp1 = _dmax(wp)+1
    exp2lrn = exploratory_learning(wp)[2:end]

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
        update_static_payoffs!(tvw, ddm, θ, i, ichar, dograd)
        ubV0 .+= discount(ddm, i, 0, θ) .* EV0_ip
        ubV1 .+= discount(ddm) .* βEV1 # β NOT baked in

        if dograd
            # dubV0 .+= discount(ddm) .* dEV0_ip
            dubV0 .+= discount(ddm, i, 0, θ) .* dEV0_ip
            dubV_ddiscount!(dubV0, EV0_ip, ddm, i, 0, θ)
            dubV1 .+= discount(ddm) .* βdEV1  # β NOT baked in
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

function solve_vf_all!(t, ddm::DDM_AbstractVF, θ, ichar, dograd; kwargs...)
    solve_vf_terminal!(   ddm,           dograd)
    solve_vf_infill!(  t, ddm, θ, ichar, dograd; kwargs...)
    learningUpdate!(   t, ddm, θ, ichar, dograd)
    solve_vf_explore!( t, ddm, θ, ichar, dograd; kwargs...)
    return nothing
end

@deprecate solve_vf_terminal!(evs,    ddm,           dograd)             solve_vf_terminal!(  ddm,           dograd)
@deprecate solve_vf_infill!(  evs, t, ddm, θ, ichar, dograd; kwargs...)  solve_vf_infill!( t, ddm, θ, ichar, dograd; kwargs...)
@deprecate learningUpdate!(   evs, t, ddm, θ, ichar, dograd)             learningUpdate!(  t, ddm, θ, ichar, dograd)
@deprecate solve_vf_explore!( evs, t, ddm, θ, ichar, dograd; kwargs...)  solve_vf_explore!(t, ddm, θ, ichar, dograd; kwargs...)
@deprecate solve_vf_all!(     evs, t, ddm, θ, ichar, dograd; kwargs...)  solve_vf_all!(    t, ddm, θ, ichar, dograd; kwargs...)

# need to reset VF to 0 for benchmarking
function solve_vf_all_timing!(t, ddm, θ, ichar, dograd; kwargs...)
    fill!(value_function(ddm), 0)
    solve_vf_all!(t, ddm, θ, ichar, dograd; kwargs...)
    return nothing
end

function solve_vf_infill_timing!(t, ddm, θ, ichar, dograd; kwargs...)
    fill!(value_function(ddm), 0)
    solve_vf_infill!(t, ddm, θ, ichar, dograd; kwargs...)
    return nothing
end

function solve_vf_and_update_itp!(ddm::DDM_AbstractVF, θ, ichar, dograd; kwargs...)
    t = DCDPTmpVars(ddm)
    solve_vf_all!(t, ddm, θ, ichar, dograd; kwargs...)
    vf = value_function(ddm)
    update_interpolation!(vf, dograd)
    return nothing
end

solve_vf_and_update_itp!(args...; kwargs...) = nothing
