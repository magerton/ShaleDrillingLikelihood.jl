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

        length(j1ptr)-1 == length(ichars) == length(j2ptr) || throw(DimensionMismatch())
        issorted(j1ptr) && issorted(j2ptr) && issorted(tptr) || throw(error("Not sorted"))
        length(jtstart) == length(tptr)-1 || throw(DimensionMismatch())
        all(jt in OneTo(length(zchars)) for jt in jtstart) || throw(DomainError(jtstart))

        return new{M,ETV,Itup}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, tchars, zchars)
    end
end

const DrillingHistoryUnit  = ObservationGroup{<:DataDrill}            # has i
const DrillingHistoryLease = ObservationGroup{<:DrillingHistoryUnit}  # has j


struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    action::Int
    state::Int
end

function Observation(d::DataDrill, i::Integer, j::Integer, t::Integer)
    0 < i <= length(d) || throw(DomainError(i))
    j in j1_range(d,i) || j == j2ptr(d,i) || throw(DomainError(j))
    t in trange(d,j)   || throw(DomainError(t))
    zt = t - tstart(d,j) + jtstart(data,j)
    return Observation(_model(d), ichars(d,i), zchars(d,zt), action, tchars(data,t)...)
end











function Observation(d::DrillingHistoryLease, t::Integer)
    j = _i(d)
    i = _i(_data(d))
    data = _data(_data(d))
    t in trange(data) || throw(DomainError())

end

function ActionSequence(data::DataDrill, i::Integer, j::Integer)
    trng = trange(data,j)
    zrng = trng .+  (jtstart(data,j)-tstart(data,j))
    @views ts = data.tchars[trng]
    @views zs = data.zchars[zrng]
    return zip(ts,zs)
end




const DataDrill = datastruct

j1ptr(   data::DataDrill) = data.j1ptr
j2ptr(   data::DataDrill) = data.j2ptr
tptr(    data::DataDrill) = data.tptr
zchars(  data::DataDrill) = data.zchars
jtstart( data::DataDrill) = data.jtstart
ichars(  data::DataDrill) = data.ichars
tchars(  data::DataDrill) = data.tchars
hasj1ptr(data::DataDrill) = length(_j1ptr(data)) > 0

length(data::DataDrill) = length(j2ptr(data))
maxj1length(data::datastruct) = hasj1ptr(data) ? maximum( diff(j1ptr(data)) ) : 1

ichars( data::DataDrill, i::Integer) = getindex(ichars(data),  i)
j1ptr(  data::DataDrill, i::Integer) = getindex(j1ptr(data),   i)
j2ptr(  data::DataDrill, i::Integer) = getindex(j2ptr(data),   i)
tptr(   data::DataDrill, j::Integer) = getindex(tptr(data),    j)
jtstart(data::DataDrill, j::Integer) = getindex(jtstart(data), j)
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

@deprecdate j2_index(data::DataDrill, i::Integer) j2ptr(data,i)
@deprecate j1_indexrange(data::DataDrill, i::Integer) j1range(data,i)
@deprecate tend(data::datastruct, j::Integer) tstop(data,j)
@deprecate ilength(data::datastruct) length(data)

# iteration through t
tstart(data::DataDrill, j::Integer) = tptr(data,j)
tstop( data::DataDrill, j::Integer) = tptr(data,j+1)-1
trange(data::DataDrill, j::Integer) = tstart(data,j) : tstop(data,j)
tlength(data::DataDrill, j::Integer) = tend(data,j) - tstart(data,j) + 1

# iteration through zchars
zcharsvec(data::DataDrill, t0::Integer) = view(zchars(data), t0:length(zchars(data)))









const ObservationGroupDrill = ObservationGroup{<:DataDrill}
const DataOrObsDrill = Union{ObservationDrill, DataDrill}
const AbstractDataStructureDrill = Union{ObservationDrill, DataDrill, ObservationGroupDrill}
