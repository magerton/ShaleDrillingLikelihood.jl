# drilling data
#----------------------------

"Drilling"
abstract type AbstractDrillModel <: AbstractModel end
struct DrillModel <: AbstractDrillModel end

struct drilling_obs <: AbstractDataStructure end

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
getindex(tv::ExogTimeVars, t::Integer) = getindex(_timevars(tv), t)
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

# Data Structs
#---------------------------

struct DataDrill{M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple} <: AbstractDataSet
    model::M

    j1ptr::Vector{Int}             # tptr, jchars is in  j1ptr[i] : j1ptr[i+1]-1
    j2ptr::UnitRange{Int}          # tptr, jchars is in  j2ptr[i]
    tptr::Vector{Int}              # tchars       is in  tptr[j] : tptr[j+1]-1
    jtstart::Vector{Int}           # zvars for lease j start at zchars[jtstart(data,j)]

    # leases per unit
    jchars::Vector{Float64}        # weights for lease observations

    # drilling histories
    ichars::Vector{ITup}
    tchars::Vector{NTuple{2,Int}}

    # time indices
    zchars::ETV

    function DataDrill(model::M, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, tchars, zchars::ETV
    ) where {M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple}

        length(j1ptr)-1 == length(ichars) == length(j2ptr)   || throw(DimensionMismatch())                     # check i
        last(j1ptr) == first(j2ptr)                          || throw(error("j1ptr and j2ptr must be conts"))  # check j
        last(j2ptr) == length(jtstart) == length(tptr)-1     || throw(DimensionMismatch())                     # check t
        issorted(j1ptr) && issorted(j2ptr) && issorted(tptr) || throw(error("Not sorted"))                     # check sort
        for j in 1:length(tptr)-1
            0 < jtstart[j] || throw(BoundsError())
            jtstart[j] + tptr[j+1] - 1 - tptr[j] < length(zchars) || throw(BoundsError())
        end
        return new{M,ETV,Itup}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, tchars, zchars)
    end
end

const DrillingHistoryUnit  = ObservationGroup{<:DataDrill}            # has i
const DrillingHistoryLease = ObservationGroup{<:DrillingHistoryUnit}  # has j
const DataDrill = datastruct

const ObservationGroupDrill = DrillingHistoryUnit
const DataOrObsDrill = Union{ObservationDrill, DataDrill}
const AbstractDataStructureDrill = Union{ObservationDrill, DataDrill, ObservationGroupDrill}

# API for DataDrill
#------------------------------------------

j1ptr(   data::DataDrill) = data.j1ptr
j2ptr(   data::DataDrill) = data.j2ptr
tptr(    data::DataDrill) = data.tptr
zchars(  data::DataDrill) = data.zchars
jtstart( data::DataDrill) = data.jtstart
ichars(  data::DataDrill) = data.ichars
tchars(  data::DataDrill) = data.tchars
j1chars( data::DataDrill) = data.j1chars
hasj1ptr(data::DataDrill) = length(_j1ptr(data)) > 0

length(     data::DataDrill) = length(j2ptr(data))
maxj1length(data::DataDrill) = hasj1ptr(data) ? maximum( diff(j1ptr(data)) ) : 1

ichars( data::DataDrill, i::Integer) = getindex(ichars(data),  i)
j1ptr(  data::DataDrill, i::Integer) = getindex(j1ptr(data),   i)
j2ptr(  data::DataDrill, i::Integer) = getindex(j2ptr(data),   i)
tptr(   data::DataDrill, j::Integer) = getindex(tptr(data),    j)
jtstart(data::DataDrill, j::Integer) = getindex(jtstart(data), j)
j1chars(data::DataDrill, j::Integer) = getindex(j1chars(data), j)
tchars( data::DataDrill, t::Integer) = getindex(tchars(data),  t)
zchars( data::DataDrill, t::Union{Integer,Date}) = getindex(zchars(data), t)


# Iteration through j1
j1start( data::DataDrill, i::Integer) = j1ptr(data,i)
j1stop(  data::DataDrill, i::Integer) = j1ptr(data,i+1)-1
j1length(data::DataDrill, i::Integer)::Int = hasj1ptr(data) ? j1stop(data,i) - j1start(data,i) + 1 : 0
j1_range(data::DataDrill, i::Integer)::UnitRange{Int}
    if hasj1ptr(data)
        return j1start(data,i) : j1stop(data,i)
    else
        return 1:0
    end
end

# iteration through t
tstart(data::DataDrill, j::Integer) = tptr(data,j)
tstop( data::DataDrill, j::Integer) = tptr(data,j+1)-1
trange(data::DataDrill, j::Integer) = tstart(data,j) : tstop(data,j)
tlength(data::DataDrill, j::Integer) = tend(data,j) - tstart(data,j) + 1

# iteration through zchars
zcharsvec(data::DataDrill, t0::Integer) = view(zchars(data), t0:length(zchars(data)))


@deprecate j2_index(data::DataDrill, i::Integer) j2ptr(data,i)
@deprecate j1_indexrange(data::DataDrill, i::Integer) j1range(data,i)
@deprecate tend(data::DataDrill, j::Integer) tstop(data,j)
@deprecate ilength(data::DataDrill) length(data)

# Iteration through layers of DataDrill
#------------------------------------------

struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    action::Int
    state::Int
end

function Observation(d::DataDrill, i::Integer, j::Integer, t::Integer)
    0 < i <= length(d) || throw(BoundsError())
    j in j1_range(d,i) || j == j2ptr(d,i) || throw(BoundsError())
    t in trange(d,j)   || throw(BoundsError())
    zt = t - tstart(d,j) + jtstart(data,j)
    action, state = tchars(data,t)
    return Observation(_model(d), ichars(d,i), zchars(d,zt), action, state)
end

# Drilling History Unit (first layer of iteration)
#------------------------------------------
DataDrill(g::DrillingHistoryUnit) = _data(g)

j1length( g::DrillingHistoryUnit) = j1length( _data(g), _i(g))
j1_range( g::DrillingHistoryUnit) = j1_range( _data(g), _i(g))
j1start(  g::DrillingHistoryUnit) = j1ptr(    _data(g), _i(g))
j1stop(   g::DrillingHistoryUnit) = j1ptr(    _data(g), _i(g)+1)-1
j2ptr(    g::DrillingHistoryUnit) = j2ptr(    _data(g), _i(g))

length(    g::DrillingHistoryUnit) = j1length(g) + 1
eachindex( g::DrillingHistoryUnit) = flatten((j1_range(g), j2ptr(g),))
firstindex(g::DrillingHistoryUnit) = j1length(g) > 0 ? j1start(g) : j2ptr(g)
lastindex( g::DrillingHistoryUnit) = j2ptr(g)

j1chars(g::DrillingHistoryUnit) = view(_j1chars(_data(g)), j1_range(g))

function iterate(g::DrillingHistoryUnit, j=firstindex(g))
    if j < firstindex(g)
        throw(BoundsError(g,j))
    elseif j > lastindex(g)
        return nothing
    else
        jp1 = j == j1stop(g) ? j2ptr(g) : j+1
        return g, jp1
    end
end

# Drilling History Lease (second layer of iteration)
#------------------------------------------
DataDrill(g::DrillingHistoryLease) = DataDrill(_data(g))

j1length( g::DrillingHistoryLease) = j1length(_data(g))
j1_range( g::DrillingHistoryLease) = j1_range(_data(g))
j2ptr(    g::DrillingHistoryLease) = j2ptr(   _data(g))
j1chars(  g::DrillingHistoryLease) = getindex(j1chars(DataDrill(g)), _i(g))

length(    g::DrillingHistoryLease) = tlength(DataDrill(g), _i(g))
eachindex( g::DrillingHistoryLease) = trange( DataDrill(g), _i(g))
firstindex(g::DrillingHistoryLease) = tstart( DataDrill(g), _i(g))
lastindex( g::DrillingHistoryLease) = tstop(  DataDrill(g), _i(g))
unitid(    g::DrillingHistoryLease) = _i(_data(g))

function iterate(g::DrillingHistoryLease, t=firstindex(g))
    if t < firstindex(g)
        throw(BoundsError(g,t))
    elseif t > lastindex(g)
        return nothing
    else
        obs = Observation(DataDrill(g), unitid(g), _i(g), t)
        return obs, t+1
    end
end
