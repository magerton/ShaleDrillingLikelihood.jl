simloglik!(grad, grp::ObservationGroupEmpty, theta, sim, dograd) = nothing
grad_simloglik!(grad, grp::ObservationGroupEmpty, theta, sim) = nothing
update!(d::Union{EmptyDataSet,ObservationGroupEmpty}, args...) = nothing


"""
    simloglik!(grad, grptup, thetas, idxs, sim, dograd; kwargs...)

Compute simulated likelihood of unit `i`

Given a tuple `grptup` of data-types (eg, royalty, pdxn, drilling), simulation 
draws `sim`, and parameter vectors `thetas` that we index into using 
an index from `idxs` that tells which elts of theta correspond to each data type
"""
function simloglik!(grad::AbstractVector, grptup::NTuple{N,ObservationGroup},
    thetas, idxs, sim::SimulationDrawsVector, dograd::Bool; kwargs...
) where {N}

    fill(_qm(sim), 0)
    logM = log(_num_sim(sim))
    zips = zip(idxs, thetas, grptup)

    for (idx,theta,grp) in zips
        @views simloglik!(grad[idx], grp, theta, sim, dograd; kwargs...)
    end

    LL = logsumexp!(_llm(sim)) - logM

    if dograd
        for (idx,theta,grp) in zips
            @views grad_simloglik!(grad[idx], grp, theta, sim)
        end
    end

    return LL
end

# SML for entire dataset
function simloglik!(grad::Vector, hess::Matrix, tmpgrad::Matrix, data::DataSetofSets,
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool; kwargs...
)
    nparm, num_i = size(tmpgrad)
    nparm == length(grad) == checksquare(hess) || throw(DimensionMismatch())

    # parameters
    thetasvw = split_thetas(data, theta)
    idxs = theta_indexes(data)

    # do updates
    rhoparm = theta_ρ(data,theta)
    update!(sim, rhoparm)
    for (dat,θ) in zip(data,thetasvw)
        update!(dat, θ)
    end

    LL = 0.0
    fill!(tmpgrad, 0)

    # iterate over observations
    for i = OneTo(num_i)
        gradi = view(tmpgrad, :, i)
        grptup = getindex.(data, i)
        simi = view(sim, i)
        fill!(_qm(sim), 0)
        LL += simloglik!(gradi, grptup, thetasvw, idxs, simi, dograd; kwargs...)
    end

    if dograd
        mul!(hess, tmpgrad, tmpgrad')
        sum!(reshape(grad, :, 1), tmpgrad)
        grad .*= -1
        hess .*= -1
    end

    return -LL
end
