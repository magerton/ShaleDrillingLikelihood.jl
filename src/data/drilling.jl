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

# DataSet
#---------------------------

export DataDrill

abstract type AbstractDataDrill <: AbstractDataSet end

struct DataDrill{M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple, XT} <: AbstractDataDrill
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
    x::Vector{XT}

    # time indices
    zchars::ETV

    function DataDrill(model::M, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, y, x::Vector{XT}, zchars::ETV
    ) where {M<:AbstractDrillModel, ETV<:ExogTimeVars, ITup<:Tuple, XT}

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
        return new{M,ETV,ITup,XT}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars)
    end
end

DataDrill(d::AbstractDataDrill) = _data(d)
DataDrill(g::AbstractDataStructure) = DataDrill(_data(g))

# What is an observation?
#------------------------------------------

struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple,XT<:Number} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    y::Int
    x::XT
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
uniti(    g::DrillUnit) = _i(g)

firstindex(grp::DrillUnit) = InitialDrilling()
lastindex( grp::DrillUnit) = DevelopmentDrilling()
length(    grp::DrillUnit) = DevelopmentDrilling()
eachindex( grp::DrillUnit) = (InitialDrilling(), DevelopmentDrilling())

# Convenience Constructors
InitialDrilling(    d::DrillUnit) = ObservationGroup(d,InitialDrilling())
DevelopmentDrilling(d::DrillUnit) = ObservationGroup(d,DevelopmentDrilling())

num_initial_leases(d::DrillUnit) = j1length(d)

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
j1chars(   g::DrillInitial) = j1chars( _data(g))

length(    g::DrillDevelopment) = 1
eachindex( g::DrillDevelopment) = j2ptr(_data(g))
firstindex(g::DrillDevelopment) = j2ptr(_data(g))
lastindex( g::DrillDevelopment) = j2ptr(_data(g))
j1chars(   g::DrillDevelopment) = 1

uniti(g::ObservationGroup) = uniti(_data(g))

# Lease (third layer of iteration)
#------------------------------------------

const DrillLease = ObservationGroup{<:AbstractDrillRegime}

length(    g::DrillLease) = tlength(DataDrill(g), _i(g))
eachindex( g::DrillLease) = trange( DataDrill(g), _i(g))
firstindex(g::DrillLease) = tstart( DataDrill(g), _i(g))
lastindex( g::DrillLease) = tstop(  DataDrill(g), _i(g))

_y(g::DrillLease) = view(_y(DataDrill(g)), eachindex(g))
_x(g::DrillLease) = view(_x(DataDrill(g)), eachindex(g))

jtstart(g::DrillLease) = jtstart(DataDrill(g), _i(g))
ztrange(g::DrillLease) = (jtstart(g)-1) .+ OneTo(length(g))
zchars( g::DrillLease) = view( zchars(DataDrill(g)), ztrange(g))
ichars( g::DrillLease) = ichars(DataDrill(g), uniti(g))
j1chars(g::DrillLease) = j1chars(DataDrill(g), _i(g))
_j(g::DrillLease) = _i(g)
_regime(g::DrillLease) = _i(_data(g))


function getindex(g::DrillLease, t)
    Observation(DataDrill(g), uniti(g), _regime(g) ,_j(g), t)
end

# Simulation of data
# -----------------------------------------------------

is_development(lease::ObservationGroup{<:DrillInitial}) = false
is_development(lease::ObservationGroup{<:DrillDevelopment}) = true

function randsumtoone(n)
    x = rand(n)
    x ./= sum(x)
    return x
end

function simulate_lease(lease::DrillLease, theta::AbstractVector{<:Number}, sim::SimulationDraw)
    m = _model(DataDrill(lease))
    length(theta) == length(m) || throw(DimensionMismatch())

    nper = length(lease)
    if nper > 0

        zc = zchars(lease)
        ic = ichars(lease)
        x = _x(lease)
        y = _y(lease)

        i = uniti(lease)
        ubv = Vector{Float64}(undef, length(actionspace(m)))

        # x[1] = initial_state(m) + is_development(lease) # FIXME

        x .= (2*is_development(lease)-1) .* abs.(x)

        for t in 1:nper
            obs = ObservationDrill(m, ic, zc[t], y[t], x[t])
            f(d) = flow(d,obs,theta,sim)
            ubv .= f.(actionspace(obs))
            logsumexp!(ubv)
            cumsum!(ubv, ubv)
            choice = searchsortedfirst(ubv, rand())-1
            y[t] = choice
            # t < nper && (x[t+1] = next_state(m,x[t],choice))
        end
    end
end

ExogTimeVarsSample(m::AbstractDrillModel, nt::Integer) = throw(error("not defined for $(m)"))
function ExogTimeVarsSample(m::TestDrillModel, nt::Integer)
    timespan = range(Date(2003,10); step=Quarter(1), length=nt)
    timevars = [(x,) for x in randn(nt)]
    return ExogTimeVars(timevars, timespan)
end

ichars_sample(m::AbstractDrillModel, num_i) = throw(error("not defined for $(m)"))
function ichars_sample(m::TestDrillModel, num_i)
    [(x,) for x in sample(0:1, num_i)]
end

xsample(d::UnivariateDistribution, nobs::Integer) = rand(d, nobs)
xsample(d::UnitRange, nobs::Integer) = sample(d, nobs)

function DataDrill(u::Vector, v::Vector, m::AbstractDrillModel, theta::AbstractVector;
    num_zt=30,
    minmaxleases::UnitRange=0:3, nperinitial::UnitRange=1:10,
    nper_development::UnitRange=0:10,
    tstart::UnitRange=5:15,
    xdomain::D=Normal()
) where {D}

    num_i = length(u)
    num_i == length(v) || throw(DimensionMismatch())
    _zchars = ExogTimeVarsSample(m, num_zt)

    # ichars
    _ichars = ichars_sample(m,num_i)

    # initial leases per unit
    initial_leases_per_unit = sample(minmaxleases, num_i)
    _jchars = vcat(collect(randsumtoone(lpu) for lpu in initial_leases_per_unit )...)
    num_initial_leases = length(_jchars)

    # observations per lease
    obs_per_lease = vcat(
        sample(nper_development, num_initial_leases),
        sample(nperinitial, num_i)
    )

    # pointers to observations
    _tptr  = 1 .+ cumsum(vcat(0, obs_per_lease))
    _j1ptr = 1 .+ cumsum(vcat(0, initial_leases_per_unit))
    _j2ptr = (last(_j1ptr)-1) .+ (1:num_i)

    nobs = last(_tptr)-1
    x = xsample(xdomain, nobs)
    y = zeros(Int, nobs)
    _jtstart = sample(tstart, num_initial_leases + num_i)

    choice_set = actionspace(m)

    data = DataDrill(m, _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)

    # update leases
    for (i,unit) in enumerate(data)
        sim = SimulationDraw(u[i], v[i], theta_drill_ρ(m,theta))
        for regimes in unit
            for lease in regimes
                simulate_lease(lease, theta, sim)
            end
        end
    end

    return data
end


function DataDrill(m, theta; num_i=100, kwargs...)
    u = randn(num_i)
    v = randn(num_i)
    DataDrill(u,v,m,theta; kwargs...)
end
