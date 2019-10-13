export ObservationDrill

# drilling data
#----------------------------

"Drilling"
abstract type AbstractDrillModel <: AbstractModel end
struct DrillModel <: AbstractDrillModel end

struct drilling_obs <: AbstractDataStructure end

# Abstract data strucutres
#---------------------------

struct ObservationDrill{M<:AbstractDrillModel} <: AbstractObservation
    model::M
    # function ObservationProduce(model::PM, y::V1, x::M, xsum::V2, nu::V1, xpnu::V2,nusum::T,nusumsq::T) where {PM<:ProductionModel,T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}}
    #     k,n = size(x)
    #     length(nu) == length(y) == n  || throw(DimensionMismatch())
    #     size(xpnu,1) == size(xsum,1) == k || throw(DimensionMismatch())
    #     return new{PM,T,V1,V2,M}(model,y,x,xsum,nu,xpnu,nusum,nusumsq)
    # end
end

struct ExogTimeVars{N, NT<:NTuple{N,AbstractVector{<:Real}}, OR<:OrdinalRange}
    timevars::NT
    timestamp::OR
    function ExogTimeVars(timevars::NT,timestamp::OR) where {N, NT<:NTuple{N,AbstractVector{<:Real}}, OR<:OrdinalRange}
        nt = length(timestamp)
        all(length.(timevars) .== nt) || throw(DimensionMismatch())
        new{N,NT,OR}(timevars,timestamp)
    end
end

_timevars(tv::ExogTimeVars) = tv.timevars
_timestamp(tv::ExogTimeVars) = tv.timestamp
getindex(tv::ExogTimeVars,t::Integer) = getindex.(_timevars(tv), t)
length(tv::ExogTimeVars) = length(_timestamp(tv))
size(tv::ExogTimeVars{N}) where {N} = length(tv), N

function DateQuarter(y::Integer, q::Integer)
    1 <= q <= 4 || throw(DomainError())
    return Date(y, 3*(q-1)+1)
end

function Quarter(i::Integer)
    i > 0 || throw(DomainError())
    return Month(3*i)
end

struct DataDrill{M<:AbstractDrillModel, T<:Real, ETV<:ExogTimeVars, ITup, R<:AbstractRange, DO<:drilling_obs} <: AbstractDataSet
    model::M

    j1ptr::Vector{Int}             # tptr, jchars is in  j1ptr[i] : j1ptr[i+1]-1
    j2ptr::UnitRange{Int}          # tptr, jchars is in  j2ptr[i]
    tptr::Vector{Int}              # tchars       is in  tptr[j] : tptr[j+1]-1
    jtstart::Vector{Int}           # zvars for lease j start at zchars[jtstart(data,j)]

    # leases per unit
    jchars::Vector{T}        # weights for lease observations

    # drilling histories
    ichars::Vector{ITup}
    tchars::Vector{DO}

    # time indices
    zchars::ETV
    zchars_time::R

    obs_ptr::Vector{Int}
    group_ptr::Vector{Int}
    # function DataProduce(
    #     model::M, y::Vector{T}, x::Matrix{T}, xsum::Matrix{T}, nu::Vector{T}, xpnu::Matrix{T},
    #     nusum::Vector{T}, nusumsq::Vector{T}, obs_ptr::Vector{Int}, group_ptr::Vector{Int}
    # ) where {M<:ProductionModel,T<:Real}
    #     k,n = size(x)
    #     length(nu) == length(y) == n  || throw(DimensionMismatch())
    #     size(xsum,1) == k || throw(DimensionMismatch())
    #     size(xpnu) == size(xsum) || throw(DimensionMismatch())
    #     issorted(obs_ptr) || throw(error("obs_ptr not sorted"))
    #     issorted(group_ptr) || throw(error("group_ptr not sorted"))
    #     length(nusum)+1 == length(nusumsq)+1 == last(group_ptr) == length(obs_ptr) || throw(DimensionMismatch("last(group_ptr)-1 != length(obs_ptr)"))
    #     last(obs_ptr)-1 == n || throw(DimensionMismatch("last(obs_ptr)-1 != length(y)"))
    #     return new{M,T}(model,y,x,xsum,nu,xpnu, nusum, nusumsq, obs_ptr,group_ptr)
    # end
end

const ObservationGroupDrill = ObservationGroup{<:DataDrill}
const DataOrObsDrill = Union{ObservationDrill, DataDrill}
const AbstractDataStructureDrill = Union{ObservationDrill, DataDrill, ObservationGroupDrill}
