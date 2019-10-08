export AbstractProductionModel, ProductionModel, ObservationProduce, ObservationGroupProduce, DataProduce

"Production"
abstract type AbstractProductionModel <: AbstractModel end
struct ProductionModel <: AbstractProductionModel end

# Abstract data strucutres
#---------------------------

struct ObservationProduce{T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}} <: AbstractObservation
    y::V1
    x::M
    xsum::V2
    nu::V1
    xpnu::V2
    nusum::T
    nusumsq::T
    function ObservationProduce(y::V1, x::M, xsum::V2, nu::V1, xpnu::V2,nusum::T,nusumsq::T) where {T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}}
        nusumsq >= 0 || throw(DomainError(nusumsq))
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xpnu,1) == size(xsum,1) == k || throw(DimensionMismatch())
        return new{T,V1,V2,M}(y,x,xsum,nu,xpnu,nusum,nusumsq)
    end
end

struct DataProduce{T<:Real} <: AbstractDataStructure
    y::Vector{T}
    x::Matrix{T}
    xsum::Matrix{T}
    nu::Vector{T}
    xpnu::Matrix{T}

    nusum::Vector{T}
    nusumsq::Vector{T}

    obs_ptr::Vector{Int}
    group_ptr::Vector{Int}
    function DataProduce(y::Vector{T}, x::Matrix{T}, xsum::Matrix{T}, nu::Vector{T}, xpnu::Matrix{T}, nusum::Vector{T}, nusumsq::Vector{T}, obs_ptr::Vector{Int}, group_ptr::Vector{Int}) where {T<:Real}
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xsum,1) == k || throw(DimensionMismatch())
        size(xpnu) == size(xsum) || throw(DimensionMismatch())
        issorted(obs_ptr) || throw(error("obs_ptr not sorted"))
        issorted(group_ptr) || throw(error("group_ptr not sorted"))
        length(nusum)+1 == length(nusumsq)+1 == last(group_ptr) == length(obs_ptr) || throw(DimensionMismatch("last(group_ptr)-1 != length(obs_ptr)"))
        last(obs_ptr)-1 == n || throw(DimensionMismatch("last(obs_ptr)-1 != length(y)"))
        return new{T}(y,x,xsum,nu,xpnu, nusum, nusumsq, obs_ptr,group_ptr)
    end
end

struct ObservationGroupProduce{T<:Real} <: AbstractObservationGroup
    data::DataProduce{T}
    i::Int
    function ObservationGroupProduce(data::DataProduce{T}, i::Int) where {T<:Real}
        1 <= i <= length(data) || throw(BoundsError(data,i))
        return new{T}(data,i)
    end
end

const DataOrObsProduction = Union{ObservationProduce,DataProduce}
const DataObsObsGrpProduction = Union{ObservationProduce,DataProduce,ObservationGroupProduce}


# Common interfaces
#---------------------------

_xpnu( d::DataOrObsProduction) = d.xpnu
_nu(   d::DataOrObsProduction) = d.nu
_y(    d::DataOrObsProduction) = d.y
_x(    d::DataOrObsProduction) = d.x
_xsum( d::DataOrObsProduction) = d.xsum
_num_x(d::DataOrObsProduction) = size(_xsum(d),1)
_nusum(d::DataOrObsProduction) = d.nusum
_nusumsq(d::DataOrObsProduction) = d.nusumsq

# Data-specific interfaces
#---------------------------

obs_ptr(  d::DataProduce) = d.obs_ptr
group_ptr(d::DataProduce) = d.group_ptr
iterate(  d::DataProduce, i::Integer=1) = i > length(d) ? nothing : (ObservationGroupProduce(d,i), i+1,)
length(   d::DataProduce) = length(group_ptr(d))-1

_num_obs(d::DataProduce) = length(obs_ptr(d))-1

groupstart( d::DataProduce, i::Integer) = getindex(group_ptr(d), i)
grouplength(d::DataProduce, i::Integer) = groupstart(d,i+1) - groupstart(d,i)
grouprange( d::DataProduce, i::Integer) = groupstart(d,i) : groupstart(d,i+1)-1

obsstart( d::DataProduce, j::Integer) = getindex(obs_ptr(d),j)
obsrange( d::DataProduce, j::Integer) = obsstart(d,j) : obsstart(d,j+1)-1
obslength(d::DataProduce, j::Integer) = obsstart(d,j+1) - obsstart(d,j)

function ObservationProduce(d::DataProduce, j::Integer)
    rng = obsrange(d,j)
    y    = view(_y(d), rng)
    nu   = view(_nu(d), rng)
    x    = view(_x(d), :, rng)
    xsum = view(_xsum(d), :, j)
    xpnu = view(_xpnu(d), :, j)
    nusum = getindex(_nusum(d), j)
    nusumsq = getindex(_nusumsq(d),j)
    return ObservationProduce(y, x, xsum, nu, xpnu, nusum, nusumsq)
end


function update_nu!(d::DataOrObsProduction, m::AbstractProductionModel, theta)
    _nu(d) .= _y(d) - _x(d)'*theta_produce_β(m,d,theta)
    let d = d
        @threads for j in OneTo(_num_obs(d))
            obs = ObservationProduce(d, j)
            nusum   = sum(_nu(obs))
            nusumsq = dot(_nu(obs), _nu(obs))
            setindex!(_nusum(d),   nusum,   j)
            setindex!(_nusumsq(d), nusumsq, j)
        end
    end
    return nothing
end

function update_xsum!(obs::ObservationProduce)
    xsum = reshape(_xsum(obs), :, 1)
    sum!(xsum, _x(obs))
    return nothing
end

function update_xpnu!(obs::ObservationProduce)
    mul!(_xpnu(obs), _x(obs), _nu(obs))
    return nothing
end

function update_over_obs(f!::Function, data::DataProduce)
    let data = data
        @threads for j in OneTo(_num_obs(data))
            f!( ObservationProduce(data, j) )
        end
    end
    return nothing
end

update_xsum!(data::DataProduce) = update_over_obs(update_xsum!, data)
update_xpnu!(data::DataProduce) = update_over_obs(update_xpnu!, data)


# Observation-specific interfaces
#---------------------------

function ==(o1::ObservationProduce, o2::ObservationProduce)
    _y(o1)==_y(o2) && _x(o1)==_x(o2) && _xsum(o1)==_xsum(o2) && _nu(o1)==_nu(o2)
end

length(o::ObservationProduce) = length(_y(o))

# Observation Group interfaces
#---------------------------

_i(        g::ObservationGroupProduce) = g.i
_data(     g::ObservationGroupProduce) = g.data
_num_x(    g::ObservationGroupProduce) = _num_x(_data(g))

length(    g::ObservationGroupProduce) = grouplength(_data(g), _i(g))
grouprange(g::ObservationGroupProduce) = grouprange( _data(g), _i(g))

obsstart( g::ObservationGroupProduce, k::Integer) = obsstart( _data(g), getindex(grouprange(g), k))
obsrange( g::ObservationGroupProduce, k::Integer) = obsrange( _data(g), getindex(grouprange(g), k))
obslength(g::ObservationGroupProduce, k::Integer) = obslength(_data(g), getindex(grouprange(g), k))

ObservationProduce(g::ObservationGroupProduce, k::Integer) = ObservationProduce(_data(g), getindex(grouprange(g), k))

iterate(g::ObservationGroupProduce, k::Integer=1) = k > length(g) ? nothing : (ObservationProduce(g,k), k+1,)


# Abstract data strucutres
#---------------------------

_num_x(m::ProductionModel, d::DataObsObsGrpProduction) = _num_x(d)
length(m::ProductionModel, d::DataObsObsGrpProduction) = _num_x(d) + 3

idx_produce_ψ(  m::ProductionModel, d) = 1
idx_produce_β(  m::ProductionModel, d) = 1 .+ (1:_num_x(d))
idx_produce_σ2η(m::ProductionModel, d) = 2 + _num_x(d)
idx_produce_σ2u(m::ProductionModel, d) = 3 + _num_x(d)

theta_produce(    m::ProductionModel, d, theta) = theta
theta_produce_ψ(  m::ProductionModel, d, theta) = theta[idx_produce_ψ(m,d)]
theta_produce_β(  m::ProductionModel, d, theta) = view(theta, idx_produce_β(m,d))
theta_produce_σ2η(m::ProductionModel, d, theta) = theta[idx_produce_σ2η(m,d)]
theta_produce_σ2u(m::ProductionModel, d, theta) = theta[idx_produce_σ2u(m,d)]

# should deprecate these?
theta_produce_ψ(  m::ProductionModel, theta) = theta[1]
theta_produce_β(  m::ProductionModel, theta) = @views theta[2:end-2]
theta_produce_σ2η(m::ProductionModel, theta) = theta[end-1]
theta_produce_σ2u(m::ProductionModel, theta) = theta[end]


# Dataset generator
#---------------------------


"""
    DataProduce(y, x, xsum, obsptr, grpptr)

Makes dataset and creats xsum, nu automatically
"""
function DataProduce(y::Vector{T}, x::Matrix{T}, obs_ptr::Vector, group_ptr::Vector) where {T}

    k,n = size(x)
    nwells = length(obs_ptr)-1

    xsum = Matrix{T}(undef, k, nwells)
    xpnu = similar(xsum)
    nu   = Vector{T}(undef, n)
    nusum = Vector{T}(undef, nwells)
    nusumsq = similar(nusum)

    data = DataProduce(y,x,xsum,nu,xpnu,nusum,nusumsq,obs_ptr,group_ptr)

    update_xsum!(data)

    return data
end




"""
    DataProduce(ngroups, maxwells, ntrange, theta)

Use `theta` to make random dataset with `ngroups` of wells that have
a random number of wells in `0:maxwells` that each have
a random draw of `ntrange` observations.
"""
function DataProduce(ngroups::Int, maxwells::Int, ntrange::UnitRange, theta::Vector)

    @assert length(theta) > 3
    alphapsi = theta[1]
    beta = theta[2:end-2]
    sigu, sigeta = theta[end-1:end]

    ncoef = length(beta)

    # groups
    psi = rand(ngroups)
    grouplens = vcat(0, collect(0:maxwells)..., sample(0:maxwells, ngroups-maxwells-1))
    groupptr = 1 .+ cumsum(grouplens)

    # wells in each group
    nwells = last(groupptr)-1

    us = rand(nwells)
    obslens = vcat(0, sample(ntrange, nwells))
    obsptr = 1 .+ cumsum(obslens)

    # observations
    nobs = last(obsptr)-1

    eta  = rand(nobs)
    x    = rand(ncoef,nobs)
    y    = x'*beta .+ sigeta .* eta

    data = DataProduce(y,x,obsptr,groupptr)

    for g in data
        i = _i(g)
        for (k,o) in enumerate(g)
            j = getindex(grouprange(g), k)
            y = _y(o)
            y .+= sigu .* us[j] .+ alphapsi .* psi[i]
        end
    end

    update_nu!(data, ProductionModel(), theta)

    return data
end
