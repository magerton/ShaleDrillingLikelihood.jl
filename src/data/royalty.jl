#= ------------------------------------------------

In general, for each type of data/eqn we have the following structure

- Data specifies
    + a `model` that tells us about extra parameters, etc
    + a vector of outcomes `y`
    + a matrix of explanatory variables `x`
    + a pre-computed vector of explanatory variables times coef, eg `xbeta` or sums like `sumx`
    + other information like the `num_choices` for multinomial/ordered probit

- Observation specifies
    + a `model`
    + an outcome, `y`
    + explanatory `x`
    + possibly pre-computed algebraic operations from bulk linear algebra obs like `xbeta = X*β` or `xpnu`
    + other info

- Other functions
    + `extra_parm(data)`: number of extra parameters
    + _nparm(data)
    + `idx_MODEL_PARAMETER(data, index)` tells the index of a particular type of parameter given the model
    + `theta_MODEL_PARAMETER(data, theta, [index])` provides the actual parameters
    + `kappa_LEVELORCUMSUM_to_CUMSUMORLEVEL(x)` transforms back & forth from ∑ᵢ̃κᵢ² ⟷ κᵢ
    + functions to simulate data
    + `coefnames(data)` to generate a list of latex coef names

# ------------------------------------------------ =#

export RoyaltyModelNoHet, RoyaltyModel, ObservationRoyalty, DataRoyalty,
    observed_choices

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
        isfinite(xbeta) || throw(DomainError(xbeta, "$xbeta = xβ = $x * β not finite")) # FIXME
        0 < y <= num_choices || throw(DomainError(y, "y = $y should be 1 <= y <= $num_choices"))
        return new{M,I,T,V}(model, y, x, xbeta, num_choices)
    end
end

struct DataRoyalty{M<:AbstractRoyaltyModel,I<:Integer,AV<:AbstractVector, T<:Real} <: AbstractDataSet
    model::M
    y::Vector{I}
    x::Matrix{T}
    xbeta::Vector{T}
    choices::AV
    function DataRoyalty(model::M, y::Vector{I}, x::Matrix{T},choices::AV) where {M,I,AV,T}
        k,n = size(x)
        length(y) == n  || throw(DimensionMismatch())
        issorted(choices) || throw(error("choices not sorted"))
        L = length(choices)
        extrema(y) == (1,L) || throw(error("lowest royalty ID > 1"))
        L == length(countmap(y)) || throw(error("highest royalty ID < L = $L"))
        return new{M,I,AV,T}(model, y, x, Vector{T}(undef, n), choices)
    end
end

const DataOrObsRoyalty = Union{ObservationRoyalty{M},DataRoyalty{M}} where {M}
const ObservationGroupRoyalty = ObservationGroup{<:DataRoyalty}
const AbstractDataStructureRoyalty = Union{ObservationRoyalty,DataRoyalty,ObservationGroupRoyalty}

function DataRoyalty(m::AbstractRoyaltyModel, d::DataRoyalty)
    DataRoyalty(m, _y(d), _x(d), choices(d))
end

# Common interfaces
#---------------------------

_xbeta(      d::DataOrObsRoyalty) = d.xbeta
choices(d::DataRoyalty) = d.choices
num_choices(d::ObservationRoyalty) = d.num_choices
num_choices(d::DataRoyalty) = length(choices(d))

observed_choices(d::DataRoyalty) = getindex(choices(d), _y(d))

@deprecate _num_choices(d::DataOrObsRoyalty) num_choices(d)

# DataRoyalty interface
#---------------------------

length(   d::DataRoyalty) = length(_y(d))
group_ptr(d::DataRoyalty) = OneTo(length(d)+1)
obs_ptr(  d::DataRoyalty) = OneTo(length(d)+1)

function choice_in_model(d::DataOrObsRoyalty, l::Integer)
    0 < l <= num_choices(d) && return true
    throw(DomainError(l, "$l outside of 1:$(num_choices(d))"))
end

function choice_in_model(d, ls)
    for l in ls
        choice_in_model(d,l)
    end
    return nothing
end

# Iteration over data
#---------------------------

function Observation(d::DataRoyalty, k::Integer)
    k ∈ eachindex(d) || throw(BoundsError(d,k))
    y = getindex(_y(d), k)
    x = view(_x(d), :, k)
    xbeta = getindex(_xbeta(d), k)
    return ObservationRoyalty(_model(d), y, x, xbeta, num_choices(d))
end

getindex(   g::ObservationGroupRoyalty, k) = Observation(_data(g), getindex(grouprange(g), k))
Observation(g::ObservationGroupRoyalty, k) = getindex(g,k)

# Observation
#---------------------------

function ==(a::ObservationRoyalty, b::ObservationRoyalty)
    _y(a) == _y(b) &&
    _x(a) == _x(b) &&
    _xbeta(a) == _xbeta(b) &&
    num_choices(a) == num_choices(b)
end

# Updating data
#---------------------------

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    # length(theta) == _num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
    mul!(_xbeta(d), _x(d)', theta)
    return nothing
end

update!(d::DataRoyalty, theta) = update_xbeta!(d,theta_royalty_β(d,theta))

# Parameter interfaces
#---------------------------

# length of RoyaltyModel parameters & gradient
extra_parm(::RoyaltyModel) = 2
extra_parm(::RoyaltyModelNoHet) = 0
extra_parm(d::DataOrObsRoyalty{<:AbstractRoyaltyModel}) = extra_parm(_model(d))

_nparm(d::DataOrObsRoyalty) = _num_x(d) + num_choices(d) - 1 + extra_parm(d)

idx_royalty(d::Union{DataOrObsRoyalty,AbstractRoyaltyModel}) = OneTo(_nparm(d))
theta_royalty(d, theta) = view(theta, idx_royalty(d))

# parameter vector
idx_royalty_ρ(d::Union{DataOrObsRoyalty{<:RoyaltyModelNoHet},RoyaltyModelNoHet}) = 1:0
idx_royalty_ψ(d::Union{DataOrObsRoyalty{<:RoyaltyModelNoHet},RoyaltyModelNoHet}) = 1:0
idx_royalty_β(d::DataOrObsRoyalty{<:RoyaltyModelNoHet}) = (1:_num_x(d))
idx_royalty_κ(d::DataOrObsRoyalty{<:RoyaltyModelNoHet}) = _num_x(d) .+ (1:num_choices(d)-1)

function idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModelNoHet}, l)
    choice_in_model(d,l)
    return _num_x(d) .+ l
end

idx_royalty_ρ(d::Union{DataOrObsRoyalty{RoyaltyModel},RoyaltyModel}) = 1
idx_royalty_ψ(d::Union{DataOrObsRoyalty{RoyaltyModel},RoyaltyModel}) = 2
idx_royalty_β(d::DataOrObsRoyalty{RoyaltyModel}) = 2 .+ (1:_num_x(d))
idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModel}) = 2 + _num_x(d) .+ (1:num_choices(d)-1)
function idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModel}, l::Integer)
    choice_in_model(d,l)
    return 2 + _num_x(d) + l
end
function idx_royalty_κ(d::DataOrObsRoyalty{RoyaltyModel}, ls::AbstractVector)
    choice_in_model(d,ls)
    return (2 + _num_x(d)) .+ ls
end


# get coefs
theta_royalty_ρ(d, theta) = theta[idx_royalty_ρ(d)]
theta_royalty_ψ(d, theta) = theta[idx_royalty_ψ(d)]
theta_royalty_β(d, theta) = view(theta, idx_royalty_β(d))
theta_royalty_κ(d, theta) = view(theta, idx_royalty_κ(d))
theta_royalty_κ(d, theta, l::Integer) = theta[idx_royalty_κ(d,l)]
theta_royalty_κ(d, theta, ls) = view(theta, idx_royalty_κ(d,ls))

kappa_level_to_cumsum(x) = vcat(first(x), sqrt.(2 .* diff(x)) )
kappa_cumsum_to_level(x) = first(x) .+ vcat(0, 0.5 .* cumsum(x[2:end].^2))
function update_kappa_level_to_cumsum!(x)
    x .= kappa_level_to_cumsum(x)
end

function theta_royalty_level_to_cumsum(d,x)
    xnew = copy(x)
    kap = theta_royalty_κ(d,x)
    xnew[idx_royalty_κ(d)] .= kappa_level_to_cumsum(kap)
    return xnew
end
function theta_royalty_cumsum_to_level(d,x)
    xnew = copy(x)
    kap = theta_royalty_κ(d,x)
    xnew[idx_royalty_κ(d)] .= kappa_cumsum_to_level(kap)
    return xnew
end


# @deprecate theta_roy(d, theta) theta_royalty(d,theta)

# check if theta is okay
theta_royalty_check(d, theta) = issorted(theta_royalty_κ(d,theta))

function check_theta(d::DataOrObsRoyalty, theta)
    _nparm(d) == length(theta) || throw(DimensionMismatch("_nparm(d) = $(_nparm(d)) != length(theta) = $(length(theta))"))
    all(isfinite.(theta)) || throw(error("theta = $theta not finite"))
    kappa = theta_royalty_κ(d,theta)
    # issorted(kappa) || @warn "kappa = $kappa not sorted"
    return true
end

function coefnames(d::DataRoyalty)
    nms = Vector{String}(undef, _nparm(d))
    if _model(d) isa RoyaltyModel
        nms[idx_royalty_ρ(d)] = "\\rho"
        nms[idx_royalty_ψ(d)] = "\\psi^0"
    end
    for (i,nmi) in enumerate(idx_royalty_β(d))
        nms[nmi] = "\\beta_$i"
    end
    for (i, nmi) in enumerate(idx_royalty_κ(d))
        nms[nmi] = "\\kappa_$i"
    end
    return nms
end


"""
    DataRoyalty(u,v,X,theta,L)

Simulate dataset for `RoyaltyModel` using `u,v` to make `ψ1`
"""
function DataRoyalty(u::AbstractVector, v::AbstractVector, X::Matrix, theta::Vector, rates::AbstractVector)

    k,nobs = size(X)
    L = length(rates)
    L >= 3 || throw(error("L = $L !>= 3"))

    k == length(theta) - (L-1) - 2 || throw(DimensionMismatch("dim mismatch b/w X: $(size(X)) and θ = $theta"))
    k >= 1 || throw(error("theta too short"))
    nobs == length(u) == length(v) || throw(DimensionMismatch())

    # get ψ1
    ψ1 = similar(u)
    dψ1dρ = similar(u)
    update_ψ1!(ψ1, u, v, first(theta))
    update_dψ1dθρ!(dψ1dρ, u, v, first(theta))

    eps    = randn(nobs)

    rstar  = theta[2] .* ψ1 .+ X'*theta[2 .+ (1:k)] .+ eps
    cutoffs = kappa_cumsum_to_level(theta[end-L+2:end])
    l = map((r) ->  searchsortedfirst(cutoffs, r), rstar)
    data = DataRoyalty(RoyaltyModel(),l,X,rates)

    return data
end

function DataRoyalty(u, v, X, theta, L::Integer=3)
    if L == 3
        rates = [1/8, 3/16, 1/4]
    else
        throw(error("L != 3"))
    end
    return DataRoyalty(u,v,X,theta,rates)
end


"""
    DataRoyalty(u,v,theta,L)

Simulate dataset for `RoyaltyModel` generating random X
"""
function DataRoyalty(u::AbstractVector, v::AbstractVector, theta::Vector, L::Integer=3)
    k = length(theta) - (L-1) - 2
    nobs = length(u)
    X = randn(k, nobs)
    return DataRoyalty(u,v,X,theta,L)
end


function DataRoyalty(num_i::Integer, theta::Vector, L::Integer=3)
    return DataRoyalty(randn(num_i), randn(num_i), theta, L)
end
