export DataDrill, DataDrillPrimitive, zchars

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

function x_in_statespace(x, wp::AbstractUnitProblem)
    setx = Set(x)
    nS = length(wp)
    if all(x .== 0)
        check = true
    else
        check = issubset( setx, 1:nS )
    end
    check || throw(error("not all x ∈ $setx are in $wp"))
    return check
end

x_in_statespace(x, wp) = true

function DataDrillChecks(j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars, wp)

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
        jtstart[j] + tptr[j+1] - 1 - tptr[j] <= length(zchars) ||
            throw(error("don't have z for all times implied by jtstart"))
    end

    x_in_statespace(x,wp)

    return true
end

struct DataDrillPrimitive{R<:AbstractStaticPayoff, ETV<:ExogTimeVars, ITup<:Tuple, XT, UP<:AbstractUnitProblem} <: AbstractDataDrill
    reward::R

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
    wp::UP

    function DataDrillPrimitive(
        reward::R, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, y, x::Vector{XT}, zchars::ETV, wp::UP
    ) where {
        R, ETV, ITup, XT, UP
    }

        chk = DataDrillChecks(j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars, wp)
        if chk == false
            throw(error("data didn't check out!"))
        end

        return new{R,ETV,ITup,XT,UP}(
            reward, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars, wp
        )
    end
end



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
    dtv::DrillingTmpVars{Float64}

    function DataDrill(model::M, j1ptr, j2ptr, tptr, jtstart,
        jchars, ichars::Vector{ITup}, y, x::Vector{XT}, zchars::ETV
    ) where {M, ETV, ITup, XT}

        wp = statespace(model)
        chk = DataDrillChecks(j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars, wp)
        if chk == false
            throw(error("data didn't check out!"))
        end

        # construct tmpvars
        dtv = DrillingTmpVars(maxj1length(j1ptr), model, Float64)

        return new{M,ETV,ITup,XT}(model, j1ptr, j2ptr, tptr, jtstart, jchars, ichars, y, x, zchars, dtv)
    end
end

DataDrill(d::DataDrill) = _data(d)
DataDrill(g::AbstractDataStructure) = DataDrill(_data(g))

function DataDrill(m::AbstractDrillModel, d::AbstractDataDrill)
    statespace(d) == statespace(m) || throw(error("state space not same"))
    return DataDrill(
        m, j1ptr(d), j2ptr(d), tptr(d), jtstart(d),
        j1chars(d), ichars(d),_y(d), _x(d), zchars(d)
    )
end

# What is an observation?
#------------------------------------------

struct ObservationDrill{M<:AbstractDrillModel,ITup<:Tuple,ZTup<:Tuple,XT<:Number} <: AbstractObservation
    model::M
    ichars::ITup
    z::ZTup
    y::Int
    x::XT
end

const ObservationDynamicDrill = ObservationDrill{<:AbstractDynamicDrillModel}
const ObservationStaticDrill = ObservationDrill{ <:AbstractStaticDrillModel}

# ichars
@inline ichars(obs::ObservationDrill) = obs.ichars
# @deprecate _ichars(obs::ObservationDrill) ichars(obs)
@inline geology(obs::ObservationDrill{<:AbstractDrillModel,<:NTuple{2,Real}}) = first(ichars(obs))
@inline royalty(obs::ObservationDrill{<:AbstractDrillModel,<:NTuple{2,Real}}) = last(ichars(obs))

# zchars
zchars(obs::ObservationDrill) = obs.z
@deprecate _z(     obs::ObservationDrill) zchars(obs)
@inline logprice(  obs::ObservationDrill) = logprice(  zchars(obs))
@inline logrigrate(obs::ObservationDrill) = logrigrate(zchars(obs))
@inline year(      obs::ObservationDrill) = year(      zchars(obs))
@inline price(     obs::ObservationDrill) = exp(logprice(obs))
@inline rigrate(   obs::ObservationDrill) = exp(logrigrate(obs))

function Observation(d::AbstractDataDrill, i::Integer, j::Integer, t::Integer)
    0 < i <= length(d) || throw(BoundsError())
    j in j1_range(d,i) || j == j2ptr(d,i) || throw(BoundsError())
    t in trange(d,j)   || throw(BoundsError())
    zt = t - tstart(d,j) + jtstart(d,j)
    return ObservationDrill(_model(d), ichars(d,i), zchars(d,zt), _y(d,t), _x(d,t))
end

Observation(d::AbstractDataDrill, i, r::AbstractRegimeType, j, t) = Observation(d,i,j,t)


# @deprecate action(obs::ObservationDrill) _y(obs)
# @deprecate state(obs::ObservationDrill) _x(obs)

# API for DataDrill
#------------------------------------------

# functions in case we want to work w/out DataDrill
hasj1ptr(   j1ptr::Vector{Int}) = length(j1ptr) > 0
maxj1length(j1ptr::Vector{Int}) = hasj1ptr(j1ptr) ? maximum( diff(j1ptr) ) : 1

# access DataDrill fields
j1ptr(   d::AbstractDataDrill) = d.j1ptr
j2ptr(   d::AbstractDataDrill) = d.j2ptr
tptr(    d::AbstractDataDrill) = d.tptr
zchars(  d::AbstractDataDrill) = d.zchars
jtstart( d::AbstractDataDrill) = d.jtstart
ichars(  d::AbstractDataDrill) = d.ichars
j1chars( d::AbstractDataDrill) = d.j1chars
DrillingTmpVars(d::DataDrill) = d.dtv
DrillingTmpVars(data) = DrillingTmpVars(_data(data))

statespace(d::DataDrillPrimitive) = d.wp
statespace(d::DataDrill) = statespace(_model(d))

coefnames(x::DataDrill)  = coefnames(_model(x))

# length of things
hasj1ptr(   d::AbstractDataDrill) = hasj1ptr(j1ptr(d))
length(     d::AbstractDataDrill) = length(j2ptr(d))
maxj1length(d::AbstractDataDrill) = maxj1length(j1ptr(d))
_nparm(     d::AbstractDataDrill) = _nparm(_model(d))
j1length(   d::AbstractDataDrill) = length(j1chars(d))
@deprecate total_leases(d) j1length(d)

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

# @deprecate j2_index(     data::DataDrill, i::Integer) j2ptr(  data,i)
# @deprecate j1_indexrange(data::DataDrill, i::Integer) j1range(data,i)
# @deprecate tend(         data::DataDrill, j::Integer) tstop(  data,j)
# @deprecate ilength(      data::DataDrill)             length( data)

# Unit (first layer of iteration)
#------------------------------------------

const DrillUnit = ObservationGroup{<:AbstractDataDrill}

j1length( g::DrillUnit) = j1length(_data(g), _i(g))
j1_range( g::DrillUnit) = j1_range(_data(g), _i(g))
j1start(  g::DrillUnit) = j1start( _data(g), _i(g))
j1stop(   g::DrillUnit) = j1stop(  _data(g), _i(g))
j2ptr(    g::DrillUnit) = j2ptr(   _data(g), _i(g))
ichars(   g::DrillUnit) = ichars(  _data(g), _i(g))
j1chars(  g::DrillUnit) = view(j1chars(_data(g)), j1_range(g))
uniti(    g::DrillUnit) = _i(g)

j1_sample(g::DrillUnit) = first(sample(j1_range(g), 1))

firstindex(grp::DrillUnit) = InitialDrilling()
lastindex( grp::DrillUnit) = DevelopmentDrilling()
length(    grp::DrillUnit) = DevelopmentDrilling()
eachindex( grp::DrillUnit) = (InitialDrilling(), DevelopmentDrilling())

# Convenience Constructors
InitialDrilling(    d::DrillUnit) = ObservationGroup(d,InitialDrilling())
DevelopmentDrilling(d::DrillUnit) = ObservationGroup(d,DevelopmentDrilling())

num_initial_leases(d::DrillUnit) = j1length(d)

function max_state(grp::DrillUnit)
    d = _data(grp)
    _t1range = tstart(d, j1start(grp)) : tstop(d, j1stop(grp))
    _t2range = trange(d, j2ptr(grp))
    x1 = length(_t1range) > 0 ? maximum(view(_x(d), _t1range)) : 0
    x2 = length(_t2range) > 0 ? maximum(view(_x(d), _t2range)) : 0
    return max(x1,x2)
end

max_states(d::DataDrill) = [max_state(g) for g in d]

total_wells_drilled(d::DataDrill) = [_D(statespace(_model(d)), max_state(g)) for g in d]


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

lease_ever_drilled(g::DrillLease) = sum(_y(g)) > 0

function lease_expired(g::DrillLease)
    m = _model(_data(g))
    wp = statespace(m)
    return exploratory_terminal(wp) in _x(g)
end



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


function initialize_x!(x, m, lease)
    x .= (2*is_development(lease)-1) .* abs.(x)
end
update_x!(x, t, m, state, d) = nothing

# check that VF is not all zeros...
check_vf_not_zero(m::AbstractStaticDrillModel) = true
check_vf_not_zero(m::AbstractDynamicDrillModel) = any(EV(value_function(m)) .!= 0)

function simulate_lease(lease::DrillLease, theta::AbstractVector{<:Number}, sim::SimulationDraw)
    m = _model(DataDrill(lease))
    length(theta) == _nparm(m) || throw(DimensionMismatch())

    tmpgrad = similar(theta)

    nper = length(lease)
    if nper > 0

        zc = zchars(lease)
        ic = ichars(lease)
        x = _x(lease)
        y = _y(lease)

        i = uniti(lease)
        ubv = Vector{Float64}(undef, length(actionspace(m)))

        initialize_x!(x, m, lease)

        dograd = false
        update_interpolation!(value_function(m), dograd)

        for t in 1:nper
            obs = ObservationDrill(m, ic, zc[t], y[t], x[t])
            f(d) = full_payoff!(tmpgrad, d, obs, theta, sim, dograd)
            actions = actionspace(obs)
            resize!(ubv, length(actions))
            ubv .= f.(actions)
            logsumexp!(ubv)
            cumsum!(ubv, ubv)
            choice = searchsortedfirst(ubv, rand())-1
            y[t] = choice
            update_x!(x, t, m, x[t], choice)
        end
    end
end

# ExogTimeVarsSample(m::AbstractDrillModel, nt::Integer) = throw(error("not defined for $(m)"))
function ExogTimeVarsSample(m::AbstractDrillModel, nt::Integer)
    timespan = range(Date(2003,10); step=Quarter(1), length=nt)
    timevars = [(x,) for x in randn(nt)]
    return ExogTimeVars(timevars, timespan)
end

# ichars_sample(m::AbstractDrillModel, num_i) = throw(error("not defined for $(m)"))
function ichars_sample(m::AbstractDrillModel, num_i)
    [(x,) for x in sample(0:1, num_i)]
end

xsample(d::UnivariateDistribution, nobs::Integer) = rand(d, nobs)
xsample(d::UnitRange, nobs::Integer) = sample(d, nobs)

function DataDrill(u::Vector, v::Vector, _zchars::ExogTimeVars, _ichars::Vector{<:NTuple{N,Number}},
    m::AbstractDrillModel, theta::AbstractVector;
    minmaxleases::UnitRange=0:3, nper_initial::UnitRange=1:10,
    nper_development::UnitRange=0:10,
    tstart::UnitRange=5:15,
    xdomain::D=Normal()
) where {D,N}

    all(u .!= v) || throw(error("u,v must be different!"))

    num_i = length(u)
    num_i == length(v) || throw(DimensionMismatch())
    num_zt = length(_zchars)

    # initial leases per unit
    initial_leases_per_unit = sample(minmaxleases, num_i)
    _jchars = vcat(collect(randsumtoone(lpu) for lpu in initial_leases_per_unit )...)
    num_initial_leases = length(_jchars)

    # observations per lease
    obs_per_lease = vcat(
        sample(nper_initial, num_initial_leases),
        sample(nper_development, num_i)
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

    θρ = theta_drill_ρ(reward(m),theta)

    # update leases
    println("Simulating $num_i units.")
    @showprogress 1 for (i,unit) in enumerate(data)
        sim = SimulationDraw(u[i], v[i], θρ)
        solve_vf_and_update_itp!(m, theta, ichars(unit), false)
        check_vf_not_zero(m) || @warn "Value Function is all 0s!"
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


function DataDrill(u, v, m, theta; num_zt=30, kwargs...)
    _zchars = ExogTimeVarsSample(m, num_zt)
    num_i = length(u)
    _ichars = ichars_sample(m,num_i)
    return DataDrill(u,v,_zchars, _ichars,m,theta;kwargs...)
end
