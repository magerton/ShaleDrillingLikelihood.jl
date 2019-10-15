# Types to define Initial vs Development Drilling
#------------------------------------------

abstract type AbstractRegimeType end
struct InitialDrilling     <: AbstractRegimeType end
struct DevelopmentDrilling <: AbstractRegimeType end
struct FinishedDrilling    <: AbstractRegimeType end

+(::InitialDrilling, i) = DevelopmentDrilling()
+(::DevelopmentDrilling, i) = FinishedDrilling()
+(::FinishedDrilling, i) = nothing

-(::InitialDrilling, i) = nothing
-(::DevelopmentDrilling, i) = InitialDrilling()
-(::FinishedDrilling, i) = DevelopmentDrilling()

==(::AbstractRegimeType, i) = false
==(::A, ::A) where {A<:AbstractRegimeType} = true

isless(::A, ::A) where {A<:AbstractRegimeType} = false
isless(::InitialDrilling,  ::Union{DevelopmentDrilling,FinishedDrilling}) = true
isless(::FinishedDrilling, ::Union{InitialDrilling,DevelopmentDrilling})  = false
isless(::DevelopmentDrilling, ::InitialDrilling)  = false
isless(::DevelopmentDrilling, ::FinishedDrilling) = true

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
    y::Vector{Int}
    x::Vector{Int}

    # time indices
    zchars::ETV

    function DataDrill(model::M, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, y, x, zchars::ETV
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

        # check tchars
        length(y) == length(x) == last(tptr)-1 ||
            throw(DimensionMismatch("lengths of y, x, and last(tptr)-1 not equal"))

        # pointers are sorted
        issorted(j1ptr) && issorted(j2ptr) && issorted(tptr) ||
            throw(error("Pointers not sorted"))

        # time vars are OK
        for j in 1:length(tptr)-1
            0 < jtstart[j] || throw(DomainError())
            jtstart[j] + tptr[j+1] - 1 - tptr[j] < length(zchars) ||
                throw(error("don't have z for all times implied by jtstart"))
        end
        return new{M,ETV,ITup}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars)
    end
end

DataDrill(d::AbstractDataDrill) = _data(d)
DataDrill(g::AbstractDataStructure) = DataDrill(_data(g))

# What is an observation?
#------------------------------------------

struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    y::Int
    x::Int
end

function Observation(d::AbstractDataDrill, i::Integer, j::Integer, t::Integer)
    0 < i <= length(d) || throw(BoundsError())
    j in j1_range(d,i) || j == j2ptr(d,i) || throw(BoundsError())
    t in trange(d,j)   || throw(BoundsError())
    zt = t - tstart(d,j) + jtstart(d,j)
    return ObservationDrill(_model(d), ichars(d,i), zchars(d,zt), _y(d,t), _x(d,t))
end

Observation(d::AbstractDataDrill, i, r::AbstractRegimeType, j, t) = Observation(d,i,j,t)

ichars(obs::ObservationDrill) = obs.ichars
zchars(obs::ObservationDrill) = obs.z
@deprecate action(obs::ObservationDrill) _y(obs)
@deprecate state(obs::ObservationDrill) _x(obs)

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
j1chars( d::DataDrill) = d.j1chars

# length
hasj1ptr(   d::AbstractDataDrill) = length(j1ptr(d)) > 0
length(     d::AbstractDataDrill) = length(j2ptr(d))
maxj1length(d::AbstractDataDrill) = hasj1ptr(d) ? maximum( diff(j1ptr(d)) ) : 1

# getindex in fields of AbstractDataDrill
ichars( d::AbstractDataDrill, i) = getindex(ichars(d),  i)
j1ptr(  d::AbstractDataDrill, i) = getindex(j1ptr(d),   i)
j2ptr(  d::AbstractDataDrill, i) = getindex(j2ptr(d),   i)

tptr(   d::AbstractDataDrill, j) = getindex(tptr(d),    j)
jtstart(d::AbstractDataDrill, j) = getindex(jtstart(d), j)
j1chars(d::AbstractDataDrill, j) = getindex(j1chars(d), j)

_y(     d::AbstractDataDrill, t) = getindex(_y(d), t)
_x(     d::AbstractDataDrill, t) = getindex(_x(d), t)
zchars( d::AbstractDataDrill, t) = getindex(zchars(d), t)

# Iteration through j1
j1start( d::AbstractDataDrill, i) = j1ptr(d,i)
j1stop(  d::AbstractDataDrill, i) = j1start(d,i+1)-1
j1length(d::AbstractDataDrill, i) = hasj1ptr(d) ? j1stop(d,i) - j1start(d,i) + 1 : 0
j1_range(d::AbstractDataDrill, i) = hasj1ptr(d) ? (j1start(d,i) : j1stop(d,i)) : (1:0)

# iteration through t
tstart( d::AbstractDataDrill, j) = tptr(d,j)
tstop(  d::AbstractDataDrill, j) = tstart(d,j+1)-1
trange( d::AbstractDataDrill, j) = tstart(d,j) : tstop(d,j)
tlength(d::AbstractDataDrill, j) = tstop(d,j) - tstart(d,j) + 1

# iteration through zchars
zcharsvec(d::AbstractDataDrill, t0) = view(zchars(d), t0:length(zchars(d)))

@deprecate j2_index(     data::DataDrill, i::Integer) j2ptr(  data,i)
@deprecate j1_indexrange(data::DataDrill, i::Integer) j1range(data,i)
@deprecate tend(         data::DataDrill, j::Integer) tstop(  data,j)
@deprecate ilength(      data::DataDrill)             length( data)

# Unit (first layer of iteration)
#------------------------------------------

const DrillUnit = ObservationGroup{<:AbstractDataDrill}

j1length( g::DrillUnit) = j1length(_data(g), _i(g))
j1_range( g::DrillUnit) = j1_range(_data(g), _i(g))
j1start(  g::DrillUnit) = j1start( _data(g), _i(g))
j1stop(   g::DrillUnit) = j1stop(  _data(g), _i(g))
j2ptr(    g::DrillUnit) = j2ptr(   _data(g), _i(g))
j1chars(  g::DrillUnit) = view(j1chars(_data(g)), j1_range(g))

firstindex(grp::DrillUnit) = InitialDrilling()
lastindex( grp::DrillUnit) = DevelopmentDrilling()
length(    grp::DrillUnit) = DevelopmentDrilling()
eachindex( grp::DrillUnit) = (InitialDrilling(), DevelopmentDrilling())

# Convenience Constructors
InitialDrilling(    d::DrillUnit) = ObservationGroup(d,InitialDrilling())
DevelopmentDrilling(d::DrillUnit) = ObservationGroup(d,DevelopmentDrilling())

# Regime (second layer of iteration)
#------------------------------------------

# At the Unit level
const AbstractDrillRegime = ObservationGroup{<:DrillUnit}
const DrillInitial        = ObservationGroup{<:DrillUnit,InitialDrilling}
const DrillDevelopment    = ObservationGroup{<:DrillUnit,DevelopmentDrilling}

length(    g::DrillInitial) = j1length(_data(g))
eachindex( g::DrillInitial) = j1_range(_data(g))
firstindex(g::DrillInitial) = j1start( _data(g))
lastindex( g::DrillInitial) = j1stop(  _data(g))

length(    g::DrillDevelopment) = 1
eachindex( g::DrillDevelopment) = j2ptr(_data(g))
firstindex(g::DrillDevelopment) = j2ptr(_data(g))
lastindex( g::DrillDevelopment) = j2ptr(_data(g))

# Lease (third layer of iteration)
#------------------------------------------

const DrillLease = ObservationGroup{<:AbstractDrillRegime}

length(    g::DrillLease) = tlength(DataDrill(g), _i(g))
eachindex( g::DrillLease) = trange( DataDrill(g), _i(g))
firstindex(g::DrillLease) = tstart( DataDrill(g), _i(g))
lastindex( g::DrillLease) = tstop(  DataDrill(g), _i(g))

function getindex(g::DrillLease, t)
    Observation(DataDrill(g), _i(_data(_data(g))), _i(_data(g)) ,_i(g), t)
end
