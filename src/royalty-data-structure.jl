export AbstractRoyaltyModel, RoyaltyModelNoHet, RoyaltyModel,
    ObservationRoyalty, DataRoyalty

# Model structure
#---------------------------

"Royalty rates"
abstract type AbstractRoyaltyModel <: AbstractModel        end
struct RoyaltyModelNoHet           <: AbstractRoyaltyModel end
struct RoyaltyModel                <: AbstractRoyaltyModel end

# Abstract data strucutres
#---------------------------

struct ObservationRoyalty{I<:Integer, T<:Real, V<:AbstractVector{T}} <: AbstractObservation
    y::I
    x::V
    xbeta::T
    num_choices::I
    function ObservationRoyalty(y::I,x::V,xbeta::T,num_choices::I) where {I<:Integer, T<:Real, V<:AbstractVector{T}}
        isfinite(xbeta) || throw(DomainError())
        0 < y <= num_choices || throw(DomainError())
        return new{I,T,V}(y, x, xbeta, num_choices)
    end
end

struct DataRoyalty{I<:Integer, T<:Real} <: AbstractDataSet
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

const DataOrObsRoyalty = Union{ObservationRoyalty,DataRoyalty}

# Common interfaces
#---------------------------

_xbeta(      d::DataOrObsRoyalty) = d.xbeta
_num_choices(d::DataOrObsRoyalty) = d.num_choices

# DataRoyalty interface
#---------------------------

group_ptr(d::DataRoyalty) = OneTo(length(_y(d))+1)
obs_ptr(  d::DataRoyalty) = OneTo(length(_y(d))+1)

function ==(a::ObservationRoyalty, b::ObservationRoyalty)
    _y(a) == _y(b) &&
    _x(a) == _x(b) &&
    _xbeta(a) == _xbeta(b) &&
    _num_choices(a) == _num_choices(b)
end

# Observation-specific interfaces
#---------------------------

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == _num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
    mul!(_xbeta(d), _x(d)', theta)
    return nothing
end

# Iteration over data
#---------------------------
function Observation(d::DataRoyalty, k::Integer)
    0 < k < length(d)+1 || throw(BoundsError(d,k))
    y = getindex(_y(d), k)
    x = view(_x(d), :, k)
    xbeta = getindex(_xbeta(d), k)
    return ObservationRoyalty(y, x, xbeta, _num_choices(d))
end

@deprecate ObservationRoyalty(d::DataRoyalty, k::Integer) Observation(d,k)

# Model / data interfaces
#---------------------------

# length of RoyaltyModel parameters & gradient
length(m::RoyaltyModelNoHet, d) = _num_x(d) + _num_choices(d) - 1
length(m::RoyaltyModel     , d) = _num_x(d) + _num_choices(d) + 1

function choice_in_model(d::DataOrObsRoyalty, l::Integer)
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
theta_roy(      m::AbstractRoyaltyModel,    theta) = theta
theta_royalty_ρ(m::AbstractRoyaltyModel, d, theta) = theta[idx_royalty_ρ(m,d)]
theta_royalty_ψ(m::AbstractRoyaltyModel, d, theta) = theta[idx_royalty_ψ(m,d)]
theta_royalty_β(m::AbstractRoyaltyModel, d, theta) = view(theta, idx_royalty_β(m,d))
theta_royalty_κ(m::AbstractRoyaltyModel, d, theta) = view(theta, idx_royalty_κ(m,d))
theta_royalty_κ(m::AbstractRoyaltyModel, d, theta, l) = theta[idx_royalty_κ(m,d,l)]

# check if theta is okay
theta_royalty_check(m::AbstractRoyaltyModel, d, theta) = issorted(theta_royalty_κ(m,d,theta))
