abstract type AbstractDataSetofSets <: AbstractDataStructure end

# import Base.Broadcast: broadcastable
export AbstractDataSetofSets, DataSetofSets, DataFull, DataRoyaltyProduce, EmptyDataSet

struct DataSetofSets{
    D<:Union{DataDrill,EmptyDataSet},
    R<:Union{DataRoyalty,EmptyDataSet},
    P<:Union{DataProduce,EmptyDataSet}
}
    data::Tuple{D,R,P}
    # drill::D
    # royalty::R
    # produce::P
    function DataSetofSets(d::D, r::R, p::P) where {D,R,P}
        drp = (d, r, p)
        lengths = length.(drp)
        issubset(lengths, (0, maximum(lengths),) ) || throw(DimensionMismatch("datasets must same or 0 lengths"))
        any(lengths .> 0 ) || throw(error("one dataset must have nonzero length"))
        return new{D,R,P}( tuple(d, r, p) )
    end
end

Broadcast.broadcastable(d::DataSetofSets) = d.data

const DataFull = DataSetofSets{<:DataDrill, <:DataRoyalty, <:DataProduce}
const DataRoyaltyProduce = DataSetofSets{EmptyDataSet, <:DataRoyalty, <:DataProduce}

drill(  d::DataSetofSets) = d.data[1] # drill
royalty(d::DataSetofSets) = d.data[2] # royalty
produce(d::DataSetofSets) = d.data[3] # produce
length(d::DataSetofSets) = 3

DataDrill(  d::DataSetofSets) = drill(d)
DataRoyalty(d::DataSetofSets) = royalty(d)
DataProduce(d::DataSetofSets) = produce(d)

idx_drill(data::DataSetofSets, coef_links) = idx_drill(DataDrill(data))

function idx_royalty(data::DataSetofSets, coef_links...)
    kd = _nparm(DataDrill(data))
    dr = DataRoyalty(data)
    return (kd-1) .+ idx_royalty(dr, coef_links...)
end

function idx_produce(data::DataSetofSets, coef_links::Vector{<:NTuple{2,Function}})
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

function thetas(data::DataSetofSets, theta::AbstractVector, coef_links...)
    d = theta_drill(  data, theta, coef_links)
    r = theta_royalty(data, theta, coef_links)
    p = theta_produce(data, theta, coef_links)
    return d, r, p
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
