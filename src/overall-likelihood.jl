function idx_theta(tup::Tuple{AbstractDataStructureRoyalty,AbstractDataStructureProduction})
    rng_roy = OneTo(_nparm(first(tup)))
    rng_pdxn = _nparm(first(tup)) .+ OneTo(_nparm(last(tup)))
    return rng_roy, rng_pdxn
end



function simloglik!(grad::AbstractVector, grptup::NTuple{N,ObservationGroup}, theta::AbstractVector, sim::SimulationDrawsVector, dograd::Bool) where {N}

    fill(_qm(sim), 0)
    logM = log(_num_sim(sim))

    idxs = idx_theta(grptup)

    for (idx,grp) in zip(idxs, grptup)
        @views simloglik!(grp, theta[idx], sim, dograd)
    end

    if !dograd
        LL = logsumexp(_llm(sim)) - logM
    else
        LL = logsumexp_and_softmax!(_llm(sim)) - logM
        for (idx,grp) in zip(idxs, grptup)
            @views grad_simloglik!(grad[idx], grp, theta[idx], sim)
        end
    end
    return LL
end


function simloglik!(grad::Vector, hess::Matrix, tmpgrad::Matrix,
    dattup::NTuple{2,AbstractDataSet},
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
)
    nparm, num_i = size(tmpgrad)
    all(length.(dattup) .== num_i) || throw(DimensionMismatch())
    nparm == length(grad) == checksquare(hess) || throw(DimensionMismatch())

    idxs = idx_theta(dattup)
    dat_roy, dat_pdxn = dattup
    theta_roy, theta_pdxn = map(idx -> view(theta, idx), idxs)

    ρ = first(theta_roy)

    # for royalty
    update_ψ1!(sim, ρ)
    update_dψ1dρ!(sim, ρ)
    update_xbeta!(dat_roy, theta_royalty_β(dat_roy, theta_roy))

    # for pdxn
    update_xsum!(dat_pdxn)
    update_nu!(dat_pdxn, theta_pdxn)
    update_xpnu!(dat_pdxn)

    LL = 0.0
    fill!(tmpgrad, 0)

    for i = OneTo(num_i)
        gradi = view(tmpgrad, :, i)
        grptup = getindex.(dattup, i)
        simi = view(sim, i)
        fill!(_qm(sim), 0)
        LL += simloglik!(gradi, grptup, theta, simi, dograd)
    end

    if dograd
        mul!(hess, tmpgrad, tmpgrad')
        sum!(reshape(grad, :, 1), tmpgrad)
        grad .*= -1
        hess .*= -1
    end

    return -LL
end
