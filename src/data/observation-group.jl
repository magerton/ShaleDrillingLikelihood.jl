export ObservationGrup

"""
Observation Groups help us iterate through a panel

Examples:
- A vector of production from a well
- The full set of leases before or after 1st well drilled
- The set of actions associated w/ 1 lease
"""
struct ObservationGroup{D<:AbstractDataStructure,I} <: AbstractObservationGroup
    data::D
    i::I
    function ObservationGroup(data::D, i::I) where {D<:AbstractDataStructure,I}
        i in eachindex(data) || throw(BoundsError(data,i))
        return new{D,I}(data,i)
    end
end

const ObservationGroupEmpty = ObservationGroup{EmptyDataSet}

# Functions for these data structures
#-----------------------

# ObservationGroup
length(o::AbstractObservation) = length(_y(o))

# default methods for iteration through an AbstractDataStructure
firstindex(d::AbstractDataStructure) = 1
lastindex( d::AbstractDataStructure) = length(d)
IndexStyle(d::AbstractDataStructure) = IndexLinear()
eachindex( d::AbstractDataStructure) = OneTo(length(d))

# Default iteration method is to create an ObservationGroup
getindex(  d::AbstractDataStructure, i) = ObservationGroup(d,i)

function iterate(d::AbstractDataStructure, i=firstindex(d))
    if i <= lastindex(d)
        return getindex(d,i), i+1
    else
        return nothing
    end
end

# AbstractDataSet iteration utilties
#------------------------------------------

_data(d::AbstractDataSet) = d

group_ptr(d::AbstractDataSet) = throw(error("group_ptr not defined for $(typeof(d))"))
obs_ptr(  d::AbstractDataSet) = throw(error("obs_ptr not defined for $(typeof(d))"))

# default method
length(   d::AbstractDataSet) = length(group_ptr(d))-1
_num_obs( d::AbstractDataSet) = length(obs_ptr(d))-1

groupstart( d::AbstractDataSet, i::Integer) = getindex(group_ptr(d), i)
groupstop(  d::AbstractDataSet, i::Integer) = groupstart(d,i+1)-1
grouplength(d::AbstractDataSet, i::Integer) = groupstop(d,i) - groupstart(d,i) + 1
grouprange( d::AbstractDataSet, i::Integer) = groupstart(d,i) : groupstop(d,i)

obsstart( d::AbstractDataSet, j::Integer) = getindex(obs_ptr(d),j)
obsstop(  d::AbstractDataSet, j::Integer) = obsstart(d,j+1)-1
obslength(d::AbstractDataSet, j::Integer) = obsstop(d,j) - obsstart(d,j) + 1
obsrange( d::AbstractDataSet, j::Integer) = obsstart(d,j) : obsstop(d,j)

# DataSet or Observation
_model(d::DataOrObs) = d.model
_y(    d::DataOrObs) = d.y
_x(    d::DataOrObs) = d.x
_num_x(d::DataOrObs) = size(_x(d), 1)

# ObservationGroup iteration utilties
#------------------------------------------

_data( g::AbstractObservationGroup) = g.data
_i(    g::AbstractObservationGroup) = g.i
_num_x(g::AbstractObservationGroup) = _num_x(_data(g))
_nparm(g::AbstractObservationGroup) = _nparm(_data(g))
_model(g::AbstractObservationGroup) = _model(_data(g))

length(    g::AbstractObservationGroup) = grouplength(_data(g), _i(g))
grouprange(g::AbstractObservationGroup) = grouprange( _data(g), _i(g))

obsstart( g::AbstractObservationGroup, k) = obsstart( _data(g), getindex(grouprange(g), k))
obsrange( g::AbstractObservationGroup, k) = obsrange( _data(g), getindex(grouprange(g), k))
obslength(g::AbstractObservationGroup, k) = obslength(_data(g), getindex(grouprange(g), k))

# Observation iteration utilties
#------------------------------------------

function update_over_obs(f!::Function, data::AbstractDataSet)
    for j in OneTo(_num_obs(data))
        f!( Observation(data, j) )
    end
    return nothing
end

# default method
update!(d::AbstractDataSet, theta) = nothing
