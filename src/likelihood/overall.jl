const DataFull = Tuple{DataDrill, DataRoyalty, DataProduce}

DataDrill(  d::DataFull) = d[1]
DataRoyalty(d::DataFull) = d[2]
DataProduce(d::DataFull) = d[3]

idx_drill(data::DataFull, coef_links) = idx_drill(DataDrill(data))

function idx_royalty(data::DataFull, coef_links...)
    kd = _nparm(DataDrill(data))
    dr = DataRoyalty(data)
    return (kd-1) .+ idx_royalty(dr, coef_links...)
end

function idx_produce(data::DataFull, coef_links::Vector{<:NTuple{2,Function}})
    kd = _nparm(DataDrill(data))
    kr = _nparm(DataRoyalty(data))
    dd = DataDrill(data)
    dp = DataProduce(data)
    idx = collect((kd+kr-1) .+ idx_produce(dp))
    for (idxp, idxd) in coef_links
        idx[idxp(dp)] = idxd(dd)
        idx[idxp(dp)+1:end] .-= 1
    end
    return idx
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
