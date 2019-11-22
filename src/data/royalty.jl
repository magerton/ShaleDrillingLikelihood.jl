export RoyaltyModelNoHet, RoyaltyModel, ObservationRoyalty, DataRoyalty

# Model structure
#---------------------------

"Royalty rates"
struct RoyaltyModelNoHet <: AbstractRoyaltyModel end
struct RoyaltyModel      <: AbstractRoyaltyModel end

# Abstract data strucutres
#---------------------------

struct ObservationRoyalty{M<:AbstractRoyaltyModel,I<:Integer, T<:Real, V<:AbstractVector{T}} <: AbstractObservation
    model::M
    y::I
    x::V
    xbeta::T
    num_choices::I
    function ObservationRoyalty(model::M,y::I,x::V,xbeta::T,num_choices::I) where {M<:AbstractRoyaltyModel,I<:Integer, T<:Real, V<:AbstractVector{T}}
        isfinite(xbeta) || throw(DomainError())
        0 < y <= num_choices || throw(DomainError())
        return new{M,I,T,V}(model, y, x, xbeta, num_choices)
    end
end

struct DataRoyalty{M<:AbstractRoyaltyModel,I<:Integer, T<:Real} <: AbstractDataSet
    model::M
    y::Vector{I}
    x::Matrix{T}
    xbeta::Vector{T}
    num_choices::I
    function DataRoyalty(model::M, y::Vector{I}, x::Matrix{T}) where {M<:AbstractRoyaltyModel,I,T}
        k,n = size(x)
        length(y) == n  || throw(DimensionMismatch())
        l,L = extrema(y)
        l == 1 || throw(error())
        L == length(countmap(y)) || throw(error())
        return new{M,I,T}(model, y, x, Vector{T}(undef, n), L)
    end
end

const DataOrObsRoyalty = Union{ObservationRoyalty{M},DataRoyalty{M}} where {M}
const ObservationGroupRoyalty = ObservationGroup{<:DataRoyalty}
const AbstractDataStructureRoyalty = Union{ObservationRoyalty,DataRoyalty,ObservationGroupRoyalty}

# Common interfaces
#---------------------------

_xbeta(      d::DataOrObsRoyalty) = d.xbeta
_num_choices(d::DataOrObsRoyalty) = d.num_choices

# DataRoyalty interface
#---------------------------

length(   d::DataRoyalty) = length(_y(d))
group_ptr(d::DataRoyalty) = OneTo(length(d)+1)
obs_ptr(  d::DataRoyalty) = OneTo(length(d)+1)

function choice_in_model(d::DataOrObsRoyalty, l::Integer)
    0 < l <= _num_choices(d) && return true
    throw(DomainError(l, "$l outside of 1:$(_num_choices(d))"))
end

# Iteration over data
#---------------------------

function Observation(d::DataRoyalty, k::Integer)
    k ∈ eachindex(d) || throw(BoundsError(d,k))
    y = getindex(_y(d), k)
    x = view(_x(d), :, k)
    xbeta = getindex(_xbeta(d), k)
    return ObservationRoyalty(_model(d), y, x, xbeta, _num_choices(d))
end

getindex(   g::ObservationGroupRoyalty, k) = Observation(_data(g), getindex(grouprange(g), k))
Observation(g::ObservationGroupRoyalty, k) = getindex(g,k)

# Observation
#---------------------------

function ==(a::ObservationRoyalty, b::ObservationRoyalty)
    _y(a) == _y(b) &&
    _x(a) == _x(b) &&
    _xbeta(a) == _xbeta(b) &&
    _num_choices(a) == _num_choices(b)
end

# Updating data
#---------------------------

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == _num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
    mul!(_xbeta(d), _x(d)', theta)
    return nothing
end

update!(d::DataRoyalty, theta) = update_xbeta!(d,theta_royalty_β(d,theta))

# Parameter interfaces
#---------------------------

# length of RoyaltyModel parameters & gradient
_nparm(d::DataOrObsRoyalty{<:RoyaltyModel})      = _num_x(d) + _num_choices(d) + 1
_nparm(d::DataOrObsRoyalty{<:RoyaltyModelNoHet}) = _num_x(d) + _num_choices(d) - 1

idx_royalty(d::Union{DataOrObsRoyalty,AbstractRoyaltyModel}) = OneTo(_nparm(d))
theta_royalty(d, theta) = view(theta, idx_royalty(d))

# parameter vector
idx_royalty_ρ(d::Union{DataOrObsRoyalty{<:RoyaltyModelNoHet},RoyaltyModelNoHet}) = 1:0
idx_royalty_ψ(d::Union{DataOrObsRoyalty{<:RoyaltyModelNoHet},RoyaltyModelNoHet}) = 1:0
idx_royalty_β(d::DataOrObsRoyalty{<:RoyaltyModelNoHet}) = (1:_num_x(d))
idx_royalty_κ(d::DataOrObsRoyalty{<:RoyaltyModelNoHet}) = _num_x(d) .+ (1:_num_choices(d)-1)

function idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModelNoHet}, l::Integer)
    choice_in_model(d,l)
    return _num_x(d) + l
end

idx_royalty_ρ(d::Union{DataOrObsRoyalty{RoyaltyModel},RoyaltyModel}) = 1
idx_royalty_ψ(d::Union{DataOrObsRoyalty{RoyaltyModel},RoyaltyModel}) = 2
idx_royalty_β(d::DataOrObsRoyalty{RoyaltyModel}) = 2 .+ (1:_num_x(d))
idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModel}) = 2 + _num_x(d) .+ (1:_num_choices(d)-1)
function idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModel}, l::Integer)
    choice_in_model(d,l)
    return 2 + _num_x(d) + l
end

# get coefs
theta_royalty_ρ(d, theta) = theta[idx_royalty_ρ(d)]
theta_royalty_ψ(d, theta) = theta[idx_royalty_ψ(d)]
theta_royalty_β(d, theta) = view(theta, idx_royalty_β(d))
theta_royalty_κ(d, theta) = view(theta, idx_royalty_κ(d))
theta_royalty_κ(d, theta, l) = theta[idx_royalty_κ(d,l)]

# @deprecate theta_roy(d, theta) theta_royalty(d,theta)

# check if theta is okay
theta_royalty_check(d, theta) = issorted(theta_royalty_κ(d,theta))


"""
    DataRoyalty(u,v,theta,L)

Simulate dataset for `RoyaltyModel` using `u,v` to make `ψ1`
"""
function DataRoyalty(u::AbstractVector, v::AbstractVector, theta::Vector, L::Integer=3)

    L >= 3 || throw(error("L = $L !>= 3"))
    k = length(theta) - (L-1) - 2
    k >= 1 || throw(error("theta too short"))
    nobs = length(u)
    nobs == length(v) || throw(DimensionMismatch())

    # get ψ1
    ψ1 = similar(u)
    dψ1dρ = similar(u)
    update_ψ1!(ψ1, u, v, first(theta))
    update_dψ1dθρ!(dψ1dρ, u, v, first(theta))

    X      = randn(k,nobs)
    eps    = randn(nobs)

    rstar  = theta[2] .* ψ1 .+ X'*theta[2 .+ (1:k)] .+ eps
    l = map((r) ->  searchsortedfirst(theta[end-L+2:end], r), rstar)
    data = DataRoyalty(RoyaltyModel(),l,X)

    return data
end

function DataRoyalty(num_i::Integer, theta::Vector, L::Integer=3)
    return DataRoyalty(randn(num_i), randn(num_i), theta, L)
end
