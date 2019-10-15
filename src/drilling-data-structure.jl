# drilling data
#----------------------------

"Drilling"
abstract type AbstractDrillModel <: AbstractModel end
struct DrillModel <: AbstractDrillModel end

# Time Variable Structs
#---------------------------

struct ExogTimeVars{N, ZTup<:NTuple{N,Real}, OR<:OrdinalRange}
    timevars::Vector{ZTup}
    timestamp::OR
    function ExogTimeVars(timevars::Vector{ZTup}, timestamp::OR) where {N, ZTup<:NTuple{N,Real}, OR<:OrdinalRange}
        length(timevars) .== length(timestamp) || throw(DimensionMismatch())
        new{N,ZTup,OR}(timevars,timestamp)
    end
end

_timevars(tv::ExogTimeVars) = tv.timevars
_timestamp(tv::ExogTimeVars) = tv.timestamp
getindex(tv::ExogTimeVars, t) = getindex(_timevars(tv), t)
function getindex(tv::ExogTimeVars, t::Date)
    ts = _timestamp(tv)
    t in ts || throw(DomainError(t))
    return getindex(_timevars(tv), searchsortedfirst(ts,t))
end
date(tv::ExogTimeVars, t::Integer)     = getindex(_timestamp(tv),t)
length(tv::ExogTimeVars) = length(_timestamp(tv))
size(tv::ExogTimeVars{N}) where {N} = length(tv), N
view(tv::ExogTimeVars, idx) = view(_timevars(tv), idx)

function DateQuarter(y::Integer, q::Integer)
    1 <= q <= 4 || throw(DomainError())
    return Date(y, 3*(q-1)+1)
end

function Quarter(i::Integer)
    i > 0 || throw(DomainError())
    return Month(3*i)
end

# What is an observation?
#------------------------------------------

struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    action::Int
    state::Int
end

function Observation(d::AbstractDataDrill, i::Integer, j::Integer, t::Integer)
    0 < i <= length(d) || throw(BoundsError())
    j in j1_range(d,i) || j == j2ptr(d,i) || throw(BoundsError())
    t in trange(d,j)   || throw(BoundsError())
    zt = t - tstart(d,j) + jtstart(d,j)
    action, state = tchars(d,t)
    return ObservationDrill(_model(d), ichars(d,i), zchars(d,zt), action, state)
end

ichars(obs::ObservationDrill) = obs.ichars
zchars(obs::ObservationDrill) = obs.z
action(obs::ObservationDrill) = obs.action
state( obs::ObservationDrill) = obs.state

# DataSet
#---------------------------

abstract type AbstractDataDrill <: AbstractDataSet end

struct DataDrill{M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple} <: AbstractDataDrill
    model::M

    j1ptr::Vector{Int}             # tptr, jchars is in  j1ptr[i] : j1ptr[i+1]-1
    j2ptr::UnitRange{Int}          # tptr, jchars is in  j2ptr[i]
    tptr::Vector{Int}              # tchars       is in  tptr[j] : tptr[j+1]-1
    jtstart::Vector{Int}           # zvars for lease j start at zchars[jtstart(data,j)]

    # leases per unit
    j1chars::Vector{Float64}       # weights for lease observations

    # drilling histories
    ichars::Vector{ITup}
    tchars::Vector{NTuple{2,Int}}

    # time indices
    zchars::ETV

    function DataDrill(model::M, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, tchars, zchars::ETV
    ) where {M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple}

        # check i
        length(j1ptr)-1 == length(ichars) == length(j2ptr)   ||
            throw(DimensionMismatch("Lengths of ichars, j2ptr, and j1ptr must agree"))

        # check j
        last(j1ptr) == first(j2ptr)                          ||
            throw(error("j1ptr and j2ptr must be conts"))

        # check t
        last(j2ptr) == length(jtstart) == length(tptr)-1     ||
            throw(DimensionMismatch("lengths of tptr, jtstart must be consistent"))

        # pointers are sorted
        issorted(j1ptr) && issorted(j2ptr) && issorted(tptr) ||
            throw(error("Pointers not sorted"))

        # time vars are OK
        for j in 1:length(tptr)-1
            0 < jtstart[j] || throw(DomainError())
            jtstart[j] + tptr[j+1] - 1 - tptr[j] < length(zchars) ||
                throw(error("don't have z for all times implied by jtstart"))
        end
        return new{M,ETV,ITup}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, tchars, zchars)
    end
end

_data(d::DataDrill) = d
DataDrill(d::DataDrill) = _data(d)

# Wrapper specifies whether we want Initial or Development drilling
#------------------------------------------

struct DataDrillInitial{M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple} <: AbstractDataDrill
    data::DataDrill{M,ETV,ITup}
end

struct DataDrillDevelopment{M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple} <: AbstractDataDrill
    data::DataDrill{M,ETV,ITup}
end

_data(d::Union{DataDrillInitial,DataDrillDevelopment}) = _data(d.data)
DataDrill(d::Union{DataDrillInitial,DataDrillDevelopment}) = _data(d)

# API for DataDrill
#------------------------------------------

# access DataDrill fields
_model(  d::DataDrill) = d.model
j1ptr(   d::DataDrill) = d.j1ptr
j2ptr(   d::DataDrill) = d.j2ptr
tptr(    d::DataDrill) = d.tptr
zchars(  d::DataDrill) = d.zchars
jtstart( d::DataDrill) = d.jtstart
ichars(  d::DataDrill) = d.ichars
tchars(  d::DataDrill) = d.tchars
j1chars( d::DataDrill) = d.j1chars


# access DataDrill fields from any AbstractDataDrill
_model(  d::AbstractDataDrill) = _model(_data(d))
j1ptr(   d::AbstractDataDrill) = j1ptr(_data(d))
j2ptr(   d::AbstractDataDrill) = j2ptr(_data(d))
tptr(    d::AbstractDataDrill) = tptr(_data(d))
zchars(  d::AbstractDataDrill) = zchars(_data(d))
jtstart( d::AbstractDataDrill) = jtstart(_data(d))
ichars(  d::AbstractDataDrill) = ichars(_data(d))
tchars(  d::AbstractDataDrill) = tchars(_data(d))
j1chars( d::AbstractDataDrill) = j1chars(_data(d))
hasj1ptr(d::AbstractDataDrill) = length(j1ptr(d)) > 0

# length
length(     data::AbstractDataDrill) = length(j2ptr(data))
maxj1length(data::AbstractDataDrill) = hasj1ptr(data) ? maximum( diff(j1ptr(data)) ) : 1

# getindex in fields of AbstractDataDrill
ichars( data::AbstractDataDrill, i::Integer) = getindex(ichars(data),  i)
j1ptr(  data::AbstractDataDrill, i::Integer) = getindex(j1ptr(data),   i)
j2ptr(  data::AbstractDataDrill, i::Integer) = getindex(j2ptr(data),   i)
tptr(   data::AbstractDataDrill, j::Integer) = getindex(tptr(data),    j)
jtstart(data::AbstractDataDrill, j::Integer) = getindex(jtstart(data), j)
j1chars(data::AbstractDataDrill, j::Integer) = getindex(j1chars(data), j)
tchars( data::AbstractDataDrill, t::Integer) = getindex(tchars(data),  t)
zchars( data::AbstractDataDrill, t::Union{Integer,Date}) = getindex(zchars(data), t)

# Iteration through j1
j1start( data::AbstractDataDrill, i::Integer) = j1ptr(data,i)
j1stop(  data::AbstractDataDrill, i::Integer) = j1ptr(data,i+1)-1
j1length(data::AbstractDataDrill, i::Integer)::Int = hasj1ptr(data) ? j1stop(data,i) - j1start(data,i) + 1 : 0
function j1_range(data::AbstractDataDrill, i::Integer)::UnitRange{Int}
    if hasj1ptr(data)
        return j1start(data,i) : j1stop(data,i)
    else
        return 1:0
    end
end

# iteration through t
tstart( data::AbstractDataDrill, j::Integer) = tptr(data,j)
tstop(  data::AbstractDataDrill, j::Integer) = tptr(data,j+1)-1
trange( data::AbstractDataDrill, j::Integer) = tstart(data,j) : tstop(data,j)
tlength(data::AbstractDataDrill, j::Integer) = tstop(data,j) - tstart(data,j) + 1

# iteration through zchars
zcharsvec(data::AbstractDataDrill, t0::Integer) = view(zchars(data), t0:length(zchars(data)))

@deprecate j2_index(data::DataDrill, i::Integer) j2ptr(data,i)
@deprecate j1_indexrange(data::DataDrill, i::Integer) j1range(data,i)
@deprecate tend(data::DataDrill, j::Integer) tstop(data,j)
@deprecate ilength(data::DataDrill) length(data)

# ObservationGroup Structures
#------------------------------------------

# At the Unit level
const AbstractDrillingHistoryUnit     = ObservationGroup{<:AbstractDataDrill}
const DrillingHistoryUnit             = ObservationGroup{<:DataDrill}
const DrillingHistoryUnit_Initial     = ObservationGroup{<:DataDrillInitial}
const DrillingHistoryUnit_Development = ObservationGroup{<:DataDrillDevelopment}
const DrillingHistoryUnit_InitOrDev   = Union{DrillingHistoryUnit_Development,DrillingHistoryUnit_Initial}

# at the lease level
const DrillingHistoryLease = ObservationGroup{<:AbstractDrillingHistoryUnit}

# either
const AbstractDrillingHistory = Union{AbstractDrillingHistoryUnit,DrillingHistoryLease}

# _data(g::AbstractDrillingHistoryUnit) already defined for ObservationGroup
DataDrill(g::AbstractDrillingHistory) = DataDrill(_data(g))

# Unit (first layer of iteration)
#------------------------------------------

j1length( g::AbstractDrillingHistoryUnit) = j1length( _data(g), _i(g))
j1_range( g::AbstractDrillingHistoryUnit) = j1_range( _data(g), _i(g))
j1start(  g::AbstractDrillingHistoryUnit) = j1ptr(    _data(g), _i(g))
j1stop(   g::AbstractDrillingHistoryUnit) = j1ptr(    _data(g), _i(g)+1)-1
j2ptr(    g::AbstractDrillingHistoryUnit) = j2ptr(    _data(g), _i(g))
j1chars(  g::AbstractDrillingHistoryUnit) = view(j1chars(_data(g)), j1_range(g))

length(    g::DrillingHistoryUnit) = j1length(g) + 1
eachindex( g::DrillingHistoryUnit) = flatten((j1_range(g), j2ptr(g),))
firstindex(g::DrillingHistoryUnit) = j1length(g) > 0 ? j1start(g) : j2ptr(g)
lastindex( g::DrillingHistoryUnit) = j2ptr(g)

length(    g::DrillingHistoryUnit_Initial) = j1length(g)
eachindex( g::DrillingHistoryUnit_Initial) = j1_range(g)
firstindex(g::DrillingHistoryUnit_Initial) = j1start(g)
lastindex( g::DrillingHistoryUnit_Initial) = j1stop(g)

length(    g::DrillingHistoryUnit_Development) = 1
eachindex( g::DrillingHistoryUnit_Development) = j2ptr(g)
firstindex(g::DrillingHistoryUnit_Development) = j2ptr(g)
lastindex( g::DrillingHistoryUnit_Development) = j2ptr(g)


function iterate(g::DrillingHistoryUnit, j::Integer=firstindex(g))
    if j < firstindex(g)
        throw(BoundsError(g,j))
    elseif j <= lastindex(g)
        jp1 = j == j1stop(g) ? j2ptr(g) : j+1
        return ObservationGroup(g,j), jp1
    else
        return nothing
    end
end

function iterate(g::DrillingHistoryUnit_InitOrDev, j::Integer=firstindex(g))
    if j < firstindex(g)
        throw(BoundsError(g,j))
    elseif j <= lastindex(g)
        return ObservationGroup(g,j), j+1
    else
        return nothing
    end
end

# Lease (second layer of iteration)
#------------------------------------------

j1length( g::DrillingHistoryLease) = j1length(_data(g))
j1_range( g::DrillingHistoryLease) = j1_range(_data(g))
j2ptr(    g::DrillingHistoryLease) = j2ptr(   _data(g))
j1chars(  g::DrillingHistoryLease) = getindex(j1chars(DataDrill(g)), _i(g))

length(    g::DrillingHistoryLease) = tlength(DataDrill(g), _i(g))
eachindex( g::DrillingHistoryLease) = trange( DataDrill(g), _i(g))
firstindex(g::DrillingHistoryLease) = tstart( DataDrill(g), _i(g))
lastindex( g::DrillingHistoryLease) = tstop(  DataDrill(g), _i(g))

function iterate(g::DrillingHistoryLease, t::Integer=firstindex(g))
    if t < firstindex(g)
        throw(BoundsError(g,t))
    elseif t > lastindex(g)
        return nothing
    else
        obs = Observation(DataDrill(g), _i(_data(g)), _i(g), t)
        return obs, t+1
    end
end
