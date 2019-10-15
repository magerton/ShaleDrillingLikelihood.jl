# For data structures
abstract type AbstractDataStructure end

"Collection of data on a particular outcome for individuals `i`"
abstract type AbstractDataSet <: AbstractDataStructure end

"What we feed into a likelihood"
abstract type AbstractObservation <: AbstractDataStructure end

const DataOrObs = Union{AbstractDataSet,AbstractObservation}

abstract type AbstractObservationGroup <: AbstractDataStructure end

"Observation Groups are associated with an individual and shock (ψ₁,ψ₂)"
struct ObservationGroup{D<:AbstractDataStructure,I} <: AbstractObservationGroup
    data::D
    i::I
    function ObservationGroup(data::D, i::I) where {D<:AbstractDataStructure,I}
        i ∈ eachindex(data) || throw(BoundsError(data,i))
        return new{D,I}(data,i)
    end
end

# AbstractDataSet iteration utilties
#------------------------------------------

group_ptr(d::AbstractDataSet) = throw(error("group_ptr not defined for $(typeof(d))"))
obs_ptr(  d::AbstractDataSet) = throw(error("obs_ptr not defined for $(typeof(d))"))

_model(d::DataOrObs) = d.model
_y(    d::DataOrObs) = d.y
_x(    d::DataOrObs) = d.x
_num_x(d::DataOrObs) = size(_x(d), 1)
_num_x(g::ObservationGroup) = _num_x(_data(g))

# observation
length(o::AbstractObservation) = length(_y(o))

# default method
length(    d::AbstractDataSet) = length(group_ptr(d))-1
_num_obs(  d::AbstractDataSet) = length(obs_ptr(d))-1
eachindex(d::AbstractDataSet) = OneTo(length(d))


getindex(  d::AbstractDataSet, i::Integer) = ObservationGroup(d,i)
iterate(   d::AbstractDataSet, i::Integer=1) = i > length(d) ? nothing : (getindex(d, i), i+1,)
firstindex(d::AbstractDataSet) = getindex(d,1)
lastindex( d::AbstractDataSet) = getindex(d,length(d))
IndexStyle(d::AbstractDataSet) = IndexLinear()
eltype(    d::AbstractDataSet) = ObservationGroup{typeof(d)}

groupstart( d::AbstractDataSet, i::Integer) = getindex(group_ptr(d), i)
grouplength(d::AbstractDataSet, i::Integer) = groupstart(d,i+1) - groupstart(d,i)
grouprange( d::AbstractDataSet, i::Integer) = groupstart(d,i) : groupstart(d,i+1)-1

obsstart( d::AbstractDataSet, j::Integer) = getindex(obs_ptr(d),j)
obsrange( d::AbstractDataSet, j::Integer) = obsstart(d,j) : obsstart(d,j+1)-1
obslength(d::AbstractDataSet, j::Integer) = obsstart(d,j+1) - obsstart(d,j)

# ObservationGroup iteration utilties
#------------------------------------------

_data( g::AbstractObservationGroup) = g.data
_i(    g::AbstractObservationGroup) = g.i
_nparm(g::AbstractObservationGroup) = _nparm(_data(g))
_model(g::AbstractObservationGroup) = _model(_data(g))

length(    g::AbstractObservationGroup) = grouplength(_data(g), _i(g))
grouprange(g::AbstractObservationGroup) = grouprange( _data(g), _i(g))

obsstart( g::AbstractObservationGroup, k) = obsstart( _data(g), getindex(grouprange(g), k))
obsrange( g::AbstractObservationGroup, k) = obsrange( _data(g), getindex(grouprange(g), k))
obslength(g::AbstractObservationGroup, k) = obslength(_data(g), getindex(grouprange(g), k))

getindex(   g::AbstractObservationGroup, k) = Observation(_data(g), getindex(grouprange(g), k))
Observation(g::AbstractObservationGroup, k) = getindex(g,k)

iterate(   g::AbstractObservationGroup, k=1) = k > length(g) ? nothing : (Observation(g,k), k+1,)
firstindex(g::AbstractObservationGroup) = 1
lastindex( g::AbstractObservationGroup) = length(g)
IndexStyle(g::AbstractObservationGroup) = IndexLinear()

# Observation iteration utilties
#------------------------------------------

function update_over_obs(f!::Function, data::AbstractDataSet)
    let data = data
        @threads for j in OneTo(_num_obs(data))
            f!( Observation(data, j) )
        end
    end
    return nothing
end
