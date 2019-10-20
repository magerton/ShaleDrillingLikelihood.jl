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

    LL = logsumexp!(_llm(sim)) - logM
    if dograd
        for (idx,grp) in zip(idxs, grptup)
            @views grad_simloglik!(grad[idx], grp, theta[idx], sim)
        end
    end
    return LL
end


function simloglik!(grad::Vector, hess::Matrix, tmpgrad::Matrix,
    dattup::Union{Tuple{DataRoyalty,DataProduce},Tuple{DataDrill,DataRoyalty,DataProduce}},
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
)
    nparm, num_i = size(tmpgrad)
    all(length.(dattup) .== num_i) || throw(DimensionMismatch())
    nparm == length(grad) == checksquare(hess) || throw(DimensionMismatch())

    # indexes for theta
    idxs = idx_theta(dattup)
    thetavws = map(idx -> view(theta, idx), idxs)

    # do updates
    ρ = first(thetavws[end-1])
    update!(sim, ρ)
    for (dat,thetavw) in zip(dattup,thetavws)
        @views update!(dat, thetavw)
    end

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
