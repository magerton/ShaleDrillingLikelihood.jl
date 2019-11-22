abstract type AbstractDataSetofSets <: AbstractDataStructure end

export AbstractDataSetofSets, DataSetofSets, DataFull, DataRoyaltyProduce, EmptyDataSet

# -------------------------------------------------
# Data structure
# -------------------------------------------------

struct DataSetofSets{D<:Union{DataDrill,EmptyDataSet},  R<:Union{DataRoyalty,EmptyDataSet}, P<:Union{DataProduce,EmptyDataSet}}
    data::Tuple{D,R,P}
    coef_links::Vector{NTuple{2,Function}}
    function DataSetofSets(d::D, r::R, p::P, coef_links) where {D,R,P}
        drp = (d, r, p)
        lengths = length.(drp)
        issubset(lengths, (0, maximum(lengths),) ) || throw(DimensionMismatch("datasets must same or 0 lengths"))
        any(lengths .> 0 ) || throw(error("one dataset must have nonzero length"))
        return new{D,R,P}( drp, coef_links)
    end
end

# If no coef_links provided
DataSetofSets(d,r,p) = DataSetofSets(d,r,p,Vector{NTuple{2,Function}}(undef,0))

# some versions
const DataFull = DataSetofSets{<:DataDrill, <:DataRoyalty, <:DataProduce}
const DataRoyaltyProduce = DataSetofSets{EmptyDataSet, <:DataRoyalty, <:DataProduce}

# -------------------------------------------------
# methods
# -------------------------------------------------

data(d::DataSetofSets) = d.data
coef_links(d::DataSetofSets) = d.coef_links
length(d::DataSetofSets) = 3

iterate(d::DataSetofSets, state...) = iterate(data(d), state...)
Broadcast.broadcastable(d::DataSetofSets) = data(d)

drill(  d::DataSetofSets) = d.data[1] # drill
royalty(d::DataSetofSets) = d.data[2] # royalty
produce(d::DataSetofSets) = d.data[3] # produce

_nparm(d::DataFull)      = sum(_nparm.(d)) - length(coef_links(d)) - 1
_nparm(d::DataSetofSets) = sum(_nparm.(d)) - length(coef_links(d))

# @deprecate DataDrill(  d::DataSetofSets) drill(d)
# @deprecate DataRoyalty(d::DataSetofSets) royalty(d)
# @deprecate DataProduce(d::DataSetofSets) produce(d)

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
idx_produce(data::DataSetofSets) = last(idx_royalty(data)) .+ idx_produce(produce(data))

# full datasets
function idx_royalty(data::DataFull)
    kd = _nparm(drill(data))
    return (kd-1) .+ idx_royalty(royalty(data))
end

function idx_produce(data::DataFull)
    dd, dr, dp = drill(data), royalty(data), produce(data)
    kd = _nparm(dd)
    kr = _nparm(dr)
    idx = collect((kd+kr-1) .+ idx_produce(dp))
    for (idxp, idxd) in coef_links(data)
        idx[idxp(dp)] = idxd(dd)
        idx[idxp(dp)+1:end] .-= 1
    end
    return idx
end

function theta_indexes(data::DataSetofSets)
    d = idx_drill(  data)
    r = idx_royalty(data)
    p = idx_produce(data)
    return d, r, p
end

# to ensure that we get a copy... NOT a view
theta_produce(d::DataFull, theta) = theta[idx_produce(d)]

function thetas(data::DataSetofSets, theta::AbstractVector)
    d = theta_drill(  data, theta)
    r = theta_royalty(data, theta)
    p = theta_produce(data, theta)
    return d, r, p
end
