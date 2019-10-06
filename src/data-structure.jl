import Base: length, size, iterate, firstindex, lastindex, getindex, IndexStyle, view, ==
using StatsBase: countmap

abstract type AbstractDataStructure end
abstract type AbstractObservationGroup end
abstract type AbstractObservation end

# ----------------------------
# royalty data
# ----------------------------

struct ObservationRoyalty{I<:Integer, T<:Real, V<:AbstractVector{T}} <: AbstractObservation
    y::I
    x::V
    xbeta::T
    num_choices::I
end

struct DataRoyalty{I<:Integer, T<:Real} <: AbstractDataStructure
    y::Vector{I}
    x::Matrix{T}
    xbeta::Vector{T}
    num_choices::I
    function DataRoyalty(y::Vector{I}, x::Matrix{T}) where {I,T}
        k,n = size(x)
        length(y) == n  || throw(DimensionMismatch())
        l,L = extrema(y)
        @assert l == 1
        @assert L == length(countmap(y))
        return new{I,T}(y, x, Vector{T}(undef, n), L)
    end
end

_y(d::Union{ObservationRoyalty, DataRoyalty}) = d.y
_x(d::Union{ObservationRoyalty, DataRoyalty}) = d.x
_xbeta(d::Union{ObservationRoyalty,DataRoyalty}) = d.xbeta
_num_choices(d::Union{ObservationRoyalty,DataRoyalty}) = d.num_choices
_choice(d::ObservationRoyalty) = _y(d)

_num_x(d::Union{ObservationRoyalty,DataRoyalty}) = size(_x(d), 1)
@deprecate _num_x(d) num_x(d)
length(d::DataRoyalty) = length(_y(d))
size(d::DataRoyalty) = length(d)

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == _num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
    mul!(_xbeta(d), _x(d)', theta)
    return nothing
end

function getindex(d::DataRoyalty, i::Integer)
    0 < i < length(d)+1 || throw(BoundsError(d,i))
    y = getindex(_y(d), i)
    x = view(_x(d), :, i)
    xbeta = getindex(_xbeta(d), i)
    return ObservationRoyalty(y, x, xbeta, _num_choices(d))
end

iterate(d::DataRoyalty) = getindex(d, 1), 2
iterate(d::DataRoyalty, i) = i > length(d) ? nothing : (getindex(d, i), i+1,)
firstindex(d::DataRoyalty) = getindex(d,1)
lastindex(d::DataRoyalty) = getindex(d,length(d))
IndexStyle(d::DataRoyalty) = IndexLinear()


# length of RoyaltyModel parameters & gradient
length(m::RoyaltyModelNoHet, d) = _num_x(d) + _num_choices(d) - 1
length(m::RoyaltyModel     , d) = _num_x(d) + _num_choices(d) + 1

function choice_in_model(d::Union{DataRoyalty,ObservationRoyalty}, l::Integer)
    0 < l <= _num_choices(d) && return true
    throw(DomainError(l, "$l outside of 1:$(_num_choices(d))"))
end

# parameter vector
idx_royalty_ρ(m::RoyaltyModelNoHet, d) = 1:0
idx_royalty_ψ(m::RoyaltyModelNoHet, d) = 1:0
idx_royalty_β(m::RoyaltyModelNoHet, d) = (1:_num_x(d))
idx_royalty_κ(m::RoyaltyModelNoHet, d) = _num_x(d) .+ (1:_num_choices(d)-1)
function idx_royalty_κ(m::RoyaltyModelNoHet, d, l::Integer)
    choice_in_model(d,l)
    return _num_x(d) + l
end

idx_royalty_ρ(m::RoyaltyModel, d) = 1
idx_royalty_ψ(m::RoyaltyModel, d) = 2
idx_royalty_β(m::RoyaltyModel, d) = 2 .+ (1:_num_x(d))
idx_royalty_κ(m::RoyaltyModel, d) = 2 + _num_x(d) .+ (1:_num_choices(d)-1)
function idx_royalty_κ(m::RoyaltyModel, d, l::Integer)
    choice_in_model(d,l)
    return 2 + _num_x(d) + l
end


# get coefs
theta_roy(      m::AbstractRoyaltyModel, theta) = theta
theta_royalty_ρ(m::AbstractRoyaltyModel, d, theta) = theta[idx_royalty_ρ(m,d)]
theta_royalty_ψ(m::AbstractRoyaltyModel, d, theta) = theta[idx_royalty_ψ(m,d)]
theta_royalty_β(m::AbstractRoyaltyModel, d, theta) = view(theta, idx_royalty_β(m,d))
theta_royalty_κ(m::AbstractRoyaltyModel, d, theta) = view(theta, idx_royalty_κ(m,d))
theta_royalty_κ(m::AbstractRoyaltyModel, d, theta, l) = theta[idx_royalty_κ(m,d,l)]

# check if theta is okay
theta_royalty_check(m::AbstractRoyaltyModel, d, theta) = issorted(theta_royalty_κ(m,d,theta))




# ----------------------------
# produce data
# ----------------------------

struct ObservationProduce{T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}} <: AbstractObservation
    y::V1
    x::M
    xsum::V2
    nu::V1
    function ObservationProduce(y::V1, x::M, xsum::V2, nu::V1) where {T<:Real, V1<:AbstractVector{T}, V2<:AbstractVector{T}, M <:AbstractMatrix{T}}
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xsum,1) == k || throw(DimensionMismatch())
        return new{T,V1,V2,M}(y,x,xsum,nu)
    end
end

struct DataProduce{T<:Real} <: AbstractDataStructure
    y::Vector{T}
    x::Matrix{T}
    xsum::Matrix{T}
    nu::Vector{T}
    obs_ptr::Vector{Int}
    group_ptr::Vector{Int}
    function DataProduce(y::Vector{T}, x::Matrix{T}, xsum::Matrix{T}, nu::Vector{T}, obs_ptr::Vector{Int}, group_ptr::Vector{Int}) where {T<:Real}
        k,n = size(x)
        length(nu) == length(y) == n  || throw(DimensionMismatch())
        size(xsum,1) == k || throw(DimensionMismatch())
        issorted(obs_ptr) || throw(error("obs_ptr not sorted"))
        issorted(group_ptr) || throw(error("group_ptr not sorted"))
        last(group_ptr) == length(obs_ptr) || throw(DimensionMismatch("last(group_ptr)-1 != length(obs_ptr)"))
        last(obs_ptr)-1 == n || throw(DimensionMismatch("last(obs_ptr)-1 != length(y)"))
        return new{T}(y,x,xsum,nu,obs_ptr,group_ptr)
    end
end

obs_ptr(  d::DataProduce) = d.obs_ptr
group_ptr(d::DataProduce) = d.group_ptr
_nu(  d::Union{ObservationProduce, DataProduce}) = d.nu
_y(   d::Union{ObservationProduce, DataProduce}) = d.y
_x(   d::Union{ObservationProduce, DataProduce}) = d.x
_xsum(d::Union{ObservationProduce, DataProduce}) = d.xsum
_num_x(d::Union{ObservationProduce, DataProduce}) = size(_xsum(d),1)

function ==(o1::ObservationProduce, o2::ObservationProduce)
    _y(o1)==_y(o2) && _x(o1)==_x(o2) && _xsum(o1)==_xsum(o2) && _nu(o1)==_nu(o2)
end

iterate(d::DataProduce, i::Integer=1) = i > length(d) ? nothing : (ObservationGroupProduce(d,i), i+1,)

length(d::DataProduce) = length(group_ptr(d))-1
length(o::ObservationProduce) = length(_y(o))

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
    return ObservationProduce(y, x, xsum, nu)
end

struct ObservationGroupProduce{T<:Real} <: AbstractObservationGroup
    data::DataProduce{T}
    i::Int
    function ObservationGroupProduce(data::DataProduce{T}, i::Int) where {T<:Real}
        1 <= i <= length(data) || throw(BoundsError(data,i))
        return new{T}(data,i)
    end
end
_i(   g::ObservationGroupProduce) = g.i
_data(g::ObservationGroupProduce) = g.data


length(    g::ObservationGroupProduce) = grouplength(_data(g), _i(g))
grouprange(g::ObservationGroupProduce) = grouprange( _data(g), _i(g))

obsstart( g::ObservationGroupProduce, k::Integer) = obsstart( _data(g), getindex(grouprange(g), k))
obsrange( g::ObservationGroupProduce, k::Integer) = obsrange( _data(g), getindex(grouprange(g), k))
obslength(g::ObservationGroupProduce, k::Integer) = obslength(_data(g), getindex(grouprange(g), k))

ObservationProduce(g::ObservationGroupProduce, k::Integer) = ObservationProduce(_data(g), getindex(grouprange(g), k))
iterate(g::ObservationGroupProduce, k::Integer=1) = k > length(g) ? nothing : (ObservationProduce(g,k), k+1,)

# ----------------------------
# drilling data
# ----------------------------

struct ObservationDrilling <: AbstractDataStructure end

# ----------------------------
# drilling data
# ----------------------------

struct DataIndividual <:AbstractDataStructure end
struct DataSet <: AbstractDataStructure end
