import Base: length, size, iterate, firstindex, lastindex, getindex, IndexStyle, view
using StatsBase: countmap

abstract type AbstractDataStructure end
abstract type AbstractObservation end

# ---------------------------------
# Simulation Draws
# ---------------------------------


const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

struct SimulationDraws{T, N, A<:AbstractArray{T,N}}
    u::A
    v::A
    function SimulationDraws(u::A, v::A) where {T<:Real,N,A<:AbstractArray{T,N}}
        size(u) == size(v)  || throw(DimensionMismatch())
        return new{T,N,A}(u,v)
    end
end

const SimulationDrawsVector{T,A} = SimulationDraws{T,1,A} where {T,A}
const SimulationDrawsMatrix{T,A} = SimulationDraws{T,2,A} where {T,A}

_u(s::SimulationDraws) = s.u
_v(s::SimulationDraws) = s.v
size(s::SimulationDraws) = size(_u(s))

function view(s::SimulationDrawsMatrix, i::Integer)
    ui = view(_u(s), :, i)
    vi = view(_v(s), :, i)
    return SimulationDraws(ui,vi)
end

getindex(s::SimulationDrawsVector, m) = getindex(_u(s), m), getindex(_v(s), m)

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

num_x(d::Union{ObservationRoyalty,DataRoyalty}) = size(_x(d), 1)
length(d::DataRoyalty) = length(_y(d))
size(d::DataRoyalty) = length(d)
function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == num_x(d) || throw(DimensionMismatch())
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

# ----------------------------
# produce data
# ----------------------------

struct ObservationProduce{T<:Real, V<:AbstractVector{T}, M <:AbstractMatrix{T}} <: AbstractObservation
    y::V
    x::M
    nu::V
end

struct DataProduce{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    y::V
    x::M
    nu::V
end

_nu(d::ObservationProduce) = d.nu
_y(d::Union{ObservationProduce, DataProduce}) = d.y
_x(d::Union{ObservationProduce, DataProduce}) = d.x

# ----------------------------
# drilling data
# ----------------------------

struct ObservationDrilling <: AbstractDataStructure
end

# ----------------------------
# drilling data
# ----------------------------

struct DataIndividual <:AbstractDataStructure
end

struct DataSet <: AbstractDataStructure
end
