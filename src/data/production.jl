export ProductionModel, ObservationProduce, ObservationGroupProduce, DataProduce

"Production"
struct ProductionModel <: AbstractProductionModel end

# Abstract data strucutres
#---------------------------

struct ObservationProduce{PM<:ProductionModel,T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}} <: AbstractObservation
    model::PM
    y::V1
    x::M
    xsum::V2
    nu::V1
    xpnu::V2
    nusum::T
    nusumsq::T
    function ObservationProduce(model::PM, y::V1, x::M, xsum::V2, nu::V1, xpnu::V2,nusum::T,nusumsq::T) where {PM<:ProductionModel,T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}}
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xpnu,1) == size(xsum,1) == k || throw(DimensionMismatch())
        return new{PM,T,V1,V2,M}(model,y,x,xsum,nu,xpnu,nusum,nusumsq)
    end
end

struct DataProduce{M<:ProductionModel,T<:Real} <: AbstractDataSet
    model::M
    y::Vector{T}
    x::Matrix{T}
    xsum::Matrix{T}
    nu::Vector{T}
    xpnu::Matrix{T}

    nusum::Vector{T}
    nusumsq::Vector{T}

    obs_ptr::Vector{Int}
    group_ptr::Vector{Int}
    function DataProduce(
        model::M, y::Vector{T}, x::Matrix{T}, xsum::Matrix{T}, nu::Vector{T}, xpnu::Matrix{T},
        nusum::Vector{T}, nusumsq::Vector{T}, obs_ptr::Vector{Int}, group_ptr::Vector{Int}
    ) where {M<:ProductionModel,T<:Real}
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xsum,1) == k || throw(DimensionMismatch())
        size(xpnu) == size(xsum) || throw(DimensionMismatch())
        issorted(obs_ptr) || throw(error("obs_ptr not sorted"))
        issorted(group_ptr) || throw(error("group_ptr not sorted"))
        length(nusum)+1 == length(nusumsq)+1 == last(group_ptr) == length(obs_ptr) || throw(DimensionMismatch("last(group_ptr)-1 != length(obs_ptr)"))
        last(obs_ptr)-1 == n || throw(DimensionMismatch("last(obs_ptr)-1 != length(y)"))
        return new{M,T}(model,y,x,xsum,nu,xpnu, nusum, nusumsq, obs_ptr,group_ptr)
    end
end

const ObservationGroupProduce = ObservationGroup{<:DataProduce}
const DataOrObsProduction = Union{ObservationProduce,DataProduce}
const AbstractDataStructureProduction = Union{ObservationProduce,DataProduce,ObservationGroupProduce}

# Common interfaces
#---------------------------

_xpnu( d::DataOrObsProduction) = d.xpnu
_nu(   d::DataOrObsProduction) = d.nu
_xsum( d::DataOrObsProduction) = d.xsum
_num_x(d::DataOrObsProduction) = size(_xsum(d),1)
_nusum(d::DataOrObsProduction) = d.nusum
_nusumsq(d::DataOrObsProduction) = d.nusumsq

# Data-specific interfaces
#---------------------------

obs_ptr(  d::DataProduce) = d.obs_ptr
group_ptr(d::DataProduce) = d.group_ptr

# Observation Group interfaces
#---------------------------

function Observation(d::DataProduce, j::Integer)
    rng  = obsrange(d,j)
    y    = view(_y(d), rng)
    nu   = view(_nu(d), rng)
    x    = view(_x(d), :, rng)
    xsum = view(_xsum(d), :, j)
    xpnu = view(_xpnu(d), :, j)
    nusum = getindex(_nusum(d), j)
    nusumsq = getindex(_nusumsq(d),j)
    return ObservationProduce(_model(d),y, x, xsum, nu, xpnu, nusum, nusumsq)
end

getindex(g::ObservationGroupProduce, k) = Observation(_data(g), getindex(grouprange(g), k))
Observation(g::ObservationGroupProduce, k) = getindex(g,k)

# Observation-specific interfaces
#---------------------------

function ==(o1::ObservationProduce, o2::ObservationProduce)
    _y(o1)       == _y(o2) &&
    _x(o1)       == _x(o2) &&
    _xsum(o1)    == _xsum(o2) &&
    _nu(o1)      == _nu(o2) &&
    _xpnu(o1)    == _xpnu(o2) &&
    _nusum(o1)   == _nusum(o2) &&
    _nusumsq(o1) == _nusumsq(o2)
end


function update_nu!(d::DataOrObsProduction, theta)
    length(theta) == _nparm(d) || throw(DimensionMismatch())
    _nu(d) .= _y(d) - _x(d)'*theta_produce_β(d,theta)
    for j in OneTo(_num_obs(d))
        obs = Observation(d, j)
        nu = _nu(obs)
        setindex!(_nusum(d),   sum(nu),    j)
        setindex!(_nusumsq(d), dot(nu,nu), j)
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

update_xsum!(data::DataProduce) = update_over_obs(update_xsum!, data)
update_xpnu!(data::DataProduce) = update_over_obs(update_xpnu!, data)

function update!(d::DataProduce, theta)
    update_nu!(d, theta)
    update_xpnu!(d)
end

# Abstract data strucutres
#---------------------------

_nparm(d::AbstractDataStructureProduction) = _num_x(d)+3
idx_produce(d::AbstractDataStructureProduction) = OneTo(_nparm(d))
theta_produce(d, theta) = view(theta, idx_produce(d))

idx_produce_ψ(  d::Union{AbstractDataStructureProduction,ProductionModel}) = 1
idx_produce_g(  d::Union{AbstractDataStructureProduction,ProductionModel}) = 2
idx_produce_t(  d::Union{AbstractDataStructureProduction,ProductionModel}) = 3
idx_produce_β(  d::AbstractDataStructureProduction) = 1 .+ (1:_num_x(d))
idx_produce_σ2η(d::AbstractDataStructureProduction) = 2 + _num_x(d)
idx_produce_σ2u(d::AbstractDataStructureProduction) = 3 + _num_x(d)

theta_produce_ψ(  d, theta) = theta[idx_produce_ψ(d)]
theta_produce_β(  d, theta) = view(theta, idx_produce_β(d))
theta_produce_σ2η(d, theta) = theta[idx_produce_σ2η(d)]
theta_produce_σ2u(d, theta) = theta[idx_produce_σ2u(d)]

theta_produce_β(  d::ProductionModel, theta) = view(theta, 2:length(theta)-2)
theta_produce_σ2η(d::ProductionModel, theta) = theta[end-1] # ACTUALLY ση
theta_produce_σ2u(d::ProductionModel, theta) = theta[end]   # ACTUALLY σu

function coefnames(d::AbstractDataStructureProduction)
    nms = Vector{String}(undef, _nparm(d))
    nms[idx_produce_ψ(d)]   = "\\alpha_\\psi"
    beta_idx = idx_produce_β(d) .- 1
    nms[idx_produce_β(d)]  .= ["\\gamma_{$i}" for i in beta_idx]
    nms[idx_produce_g(d)]   = "\\alpha_g"
    nms[idx_produce_σ2η(d)] = "\\sigma_\\eta"
    nms[idx_produce_σ2u(d)] = "\\sigma_u"
    return nms
end

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

    data = DataProduce(ProductionModel(),y,x,xsum,nu,xpnu,nusum,nusumsq,obs_ptr,group_ptr)

    update_xsum!(data)

    return data
end




"""
    DataProduce(ngroups, maxwells, ntrange, theta)

Use `theta` to make random dataset with `ngroups` of wells that have
a random number of wells in `0:maxwells` that each have
a random draw of `ntrange` observations.
"""
function DataProduce(psi::Vector, maxwells::Int, ntrange::UnitRange, theta::Vector, args...)
    ngroups = length(psi)
    grouplens = vcat(collect(0:maxwells)..., sample(0:maxwells, ngroups-maxwells-1))
    ncoef = length(theta)-3

    return DataProduce(psi,grouplens,ntrange,theta, args...)
end


function DataProduce(psi::Vector, grouplens::Vector, ntrange::UnitRange, theta::Vector, ivars::Union{Number,Vector}=0)

    length(psi) == length(grouplens) || throw(DimensionMismatch())
    length(theta) > 3 || throw(DimensionMismatch())
    alphapsi = theta[1]
    beta = theta[2:end-2]
    sigeta, sigu = theta[end-1:end]

    ncoef = length(beta)

    # groups
    groupptr = 1 .+ cumsum(vcat(0, grouplens))
    nwells = sum(grouplens)

    us = randn(nwells)
    obslens = vcat(0, sample(ntrange, nwells))
    obsptr = 1 .+ cumsum(obslens)

    # observations
    nobs = last(obsptr)-1

    eta  = randn(nobs)
    x    = randn(ncoef,nobs)
    y    = similar(eta)

    data = DataProduce(y,x,obsptr,groupptr)

    if length(ivars) == length(psi)
        for (i,g) in enumerate(data)
            for (k,o) in enumerate(g)
                j = getindex(grouprange(g), k)
                xj = _x(o)
                xj[1,:] .= ivars[i]
            end
        end
    end

    y .= x'*beta .+ sigeta .* eta

    for (i,g) in enumerate(data)
        for (k,o) in enumerate(g)
            j = getindex(grouprange(g), k)
            y = _y(o) # if we don't do this, errors on julia v1.1.1
            y .+= sigu .* us[j] .+ alphapsi .* psi[i]
        end
    end

    update_xsum!(data)
    update_nu!(data, theta)

    return data
end

DataProduce(ngroups::Int, args...) = DataProduce(randn(ngroups), args...)
