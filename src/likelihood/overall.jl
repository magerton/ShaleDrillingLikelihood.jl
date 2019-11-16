simloglik!(grp::ObservationGroupEmpty, theta, sim, dograd) = nothing
grad_simloglik!(grad, grp::ObservationGroupEmpty, theta, sim) = nothing


function simloglik!(grad::AbstractVector, grptup::NTuple{N,ObservationGroup}, thetas, idxs, sim::SimulationDrawsVector, dograd::Bool) where {N}

    fill(_qm(sim), 0)
    logM = log(_num_sim(sim))

    for (grp, theta) in zip(grptup, thetas)
        @views simloglik!(grp, theta, sim, dograd)
    end

    LL = logsumexp!(_llm(sim)) - logM

    if dograd
        for (idx,theta,grp) in zip(idxs, thetas, grptup)
            @views grad_simloglik!(grad[idx], grp, theta, sim)
        end
    end

    return LL
end


function simloglik!(grad::Vector, hess::Matrix, tmpgrad::Matrix, data::DataSetofSets,
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
)
    nparm, num_i = size(tmpgrad)
    nparm == length(grad) == checksquare(hess) || throw(DimensionMismatch())

    # parameters
    thetasvw = thetas(data, theta)
    idxs = theta_indexes(data)

    # do updates
    ρ = first(thetasvw[end-1])
    update!(sim, ρ)
    for (dat,theta) in zip(data,thetasvw)
        update!(dat, theta)
    end

    LL = 0.0
    fill!(tmpgrad, 0)

    # iterate over observations
    for i = OneTo(num_i)
        gradi = view(tmpgrad, :, i)
        grptup = getindex.(data, i)
        simi = view(sim, i)
        fill!(_qm(sim), 0)
        LL += simloglik!(gradi, grptup, thetasvw, idxs, simi, dograd)
    end

    if dograd
        mul!(hess, tmpgrad, tmpgrad')
        sum!(reshape(grad, :, 1), tmpgrad)
        grad .*= -1
        hess .*= -1
    end

    return -LL
end
