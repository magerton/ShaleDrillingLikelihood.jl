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

const DataOrObsRoyalty = Union{ObservationRoyalty,DataRoyalty}

# Common interfaces
#---------------------------

_y(          d::DataOrObsRoyalty) = d.y
_x(          d::DataOrObsRoyalty) = d.x
_xbeta(      d::DataOrObsRoyalty) = d.xbeta
_num_choices(d::DataOrObsRoyalty) = d.num_choices
_num_x(      d::DataOrObsRoyalty) = size(_x(d), 1)

# Observation-specific interfaces
#---------------------------
length(d::DataRoyalty) = length(_y(d))
size(  d::DataRoyalty) = length(d)

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == _num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
    mul!(_xbeta(d), _x(d)', theta)
    return nothing
end

# Iteration over data
#---------------------------
function getindex(d::DataRoyalty, i::Integer)
    0 < i < length(d)+1 || throw(BoundsError(d,i))
    y = getindex(_y(d), i)
    x = view(_x(d), :, i)
    xbeta = getindex(_xbeta(d), i)
    return ObservationRoyalty(y, x, xbeta, _num_choices(d))
end

iterate(   d::DataRoyalty, i=1) = i > length(d) ? nothing : (getindex(d, i), i+1,)
firstindex(d::DataRoyalty) = getindex(d,1)
lastindex( d::DataRoyalty) = getindex(d,length(d))
IndexStyle(d::DataRoyalty) = IndexLinear()


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
