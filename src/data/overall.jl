abstract type AbstractDataSetofSets <: AbstractDataStructure end

export AbstractDataSetofSets,
    DataSetofSets,
    DataFull,
    DataRoyaltyProduce,
    EmptyDataSet,
    DataDrillOnly,
    theta_linked

# -------------------------------------------------
# Data structure
# -------------------------------------------------

function check_coefs_in_nparm(cf, d::AbstractDataSet)
    n = _nparm(d)
    issubset(Set(cf), OneTo(n)) || throw(error("coefs $cf not in 1:$n"))
    allunique(cf) || throw(error("coefs $cf not unique"))
    return true
end

struct DataSetofSets{
    D<:Union{AbstractDataDrill,EmptyDataSet},  R<:Union{DataRoyalty,EmptyDataSet}, P<:Union{DataProduce,EmptyDataSet},
    }
    data::Tuple{D,R,P}
    coef_link_pdxn::Vector{Int}
    coef_link_drill::Vector{Int}
    function DataSetofSets(d::D, r::R, p::P, cfp::V, cfd::V) where {D,R,P,V<:Vector{Int}}
        drp = (d, r, p)
        lengths = length.(drp)
        issubset(lengths, (0, maximum(lengths),) ) || throw(DimensionMismatch("datasets must same or 0 lengths"))
        length(cfp) == length(cfd) || throw(DimensionMismatch())
        check_coefs_in_nparm(cfp, p)
        check_coefs_in_nparm(cfd, d)
        any(lengths .> 0 ) || throw(error("one dataset must have nonzero length"))
        return new{D,R,P}(drp, cfp, cfd)
    end
end

# If no coef_links provided
DataSetofSets(d,r,p) = DataSetofSets(d,r,p,zeros(Int,0),zeros(Int,0))

function DataSetofSets(d,r,p, cfp::Vector, cfd::Vector)
    DataSetofSets(d,r,p, map(f -> f(p), cfp), map(f -> f(d), cfd))
end

function DataSetofSets(d,r,p,cfl::Vector{<:Tuple})
    return DataSetofSets(d,r,p, first.(cfl), last.(cfl))
end

function SimulationDraws(data::DataSetofSets, M)
    n = num_i(data)
    k = _nparm(drill(data))
    return SimulationDraws(M,n,k)
end


# some versions
const DataFull = DataSetofSets{<:AbstractDataDrill, <:DataRoyalty, <:DataProduce}
const DataRoyaltyProduce = DataSetofSets{EmptyDataSet, <:DataRoyalty, <:DataProduce}
const DataDrillOnly = DataSetofSets{<:AbstractDataDrill, EmptyDataSet, EmptyDataSet}

# -------------------------------------------------
# methods
# -------------------------------------------------

data(d::DataSetofSets) = d.data
coef_link_drill(d::DataSetofSets) = d.coef_link_drill
coef_link_pdxn(d::DataSetofSets) = d.coef_link_pdxn
coef_links(d::DataSetofSets) = zip(coef_link_pdxn(d), coef_link_drill(d))
length(d::DataSetofSets) = 3
num_i(d::DataSetofSets) = maximum(length.(d))

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

# theta_ρ(data::DataSetofSets{EmptyDataSet,<:DataRoyalty}, theta) = theta[1]
theta_ρ(data::DataSetofSets{<:AbstractDataDrill}, theta) = theta[_nparm(drill(data))]
theta_ρ(data::DataRoyaltyProduce, theta) = theta[1]


function theta_linked(thetas::NTuple{3,AbstractVector}, data::DataFull)
    length.(thetas) == _nparm.(data) || throw(DimensionMismatch())
    thet_d, thet_r, thet_p = thetas

    data_p = produce(data)
    idx_p = OneTo(_nparm(data_p))
    idx_p_drops = coef_link_pdxn(data)

    thet_p_short = [thet_p[i] for i in idx_p if i ∉ idx_p_drops]

    return vcat(thet_d, thet_r[2:end], thet_p_short)
end


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
        idx[idxp] = idxd
        idx[idxp+1:end] .-= 1
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
