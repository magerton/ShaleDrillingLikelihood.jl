abstract type AbstractDataSetofSets <: AbstractDataStructure end

export AbstractDataSetofSets, DataSetofSets, DataFull, DataRoyaltyProduce, EmptyDataSet

# -------------------------------------------------
# Data structure
# -------------------------------------------------

struct DataSetofSets{D<:Union{DataDrill,EmptyDataSet},  R<:Union{DataRoyalty,EmptyDataSet}, P<:Union{DataProduce,EmptyDataSet}}
    data::Tuple{D,R,P}
    function DataSetofSets(d::D, r::R, p::P) where {D,R,P}
        drp = (d, r, p)
        lengths = length.(drp)
        issubset(lengths, (0, maximum(lengths),) ) || throw(DimensionMismatch("datasets must same or 0 lengths"))
        any(lengths .> 0 ) || throw(error("one dataset must have nonzero length"))
        return new{D,R,P}( tuple(d, r, p) )
    end
end

data(d::DataSetofSets) = d.data
iterate(d::DataSetofSets, state...) = iterate(data(d), state...)
Broadcast.broadcastable(d::DataSetofSets) = data(d)

const DataFull = DataSetofSets{<:DataDrill, <:DataRoyalty, <:DataProduce}
const DataRoyaltyProduce = DataSetofSets{EmptyDataSet, <:DataRoyalty, <:DataProduce}

drill(  d::DataSetofSets) = d.data[1] # drill
royalty(d::DataSetofSets) = d.data[2] # royalty
produce(d::DataSetofSets) = d.data[3] # produce
length(d::DataSetofSets) = 3

@deprecate DataDrill(  d::DataSetofSets) drill(d)
@deprecate DataRoyalty(d::DataSetofSets) royalty(d)
@deprecate DataProduce(d::DataSetofSets) produce(d)

# -------------------------------------------------
# Indexing into parameter vector
# -------------------------------------------------

# if dataset is empty
idx_drill(  ::EmptyDataSet) = 1:0
idx_royalty(::EmptyDataSet) = 1:0
idx_produce(::EmptyDataSet) = 1:0

# standard indexing, no overlap
idx_drill(  data::DataSetofSets) = idx_drill(drill(data))
idx_royalty(data::DataSetofSets) = last(idx_drill(data)) .+ idx_royalty(royalty(data))
idx_produce(data::DataSetofSets, coef_links) = last(idx_royalty(data)) .+ idx_produce(produce(data))

_nparm(d::DataFull, coef_links=[]) = sum(_nparm.(d))-1-length(coef_links)
_nparm(d::DataSetofSets, coef_links=[]) = sum(_nparm.(d) )

# full datasets
function idx_royalty(data::DataFull)
    kd = _nparm(drill(data))
    dr = royalty(data)
    return (kd-1) .+ idx_royalty(dr)
end

function idx_produce(data::DataFull, coef_links::Vector{<:Tuple})
    dd, dr, dp = drill(data), royalty(data), produce(data)
    kd = _nparm(dd)
    kr = _nparm(dr)
    idx = collect((kd+kr-1) .+ idx_produce(dp))
    for (idxp, idxd) in coef_links
        idx[idxp(dp)] = idxd(dd)
        idx[idxp(dp)+1:end] .-= 1
    end
    return idx
end

function theta_indexes(data::DataSetofSets, coef_links=[])
    d = idx_drill(  data)
    r = idx_royalty(data)
    p = idx_produce(data, coef_links)
    return d, r, p
end

function thetas(data::DataSetofSets, theta::AbstractVector, coef_links=[])
    d = theta_drill(  data, theta)
    r = theta_royalty(data, theta)
    p = theta_produce(data, theta, coef_links)
    return d, r, p
end
