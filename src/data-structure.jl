import Base: length, size, iterate, firstindex, lastindex, getindex, IndexStyle, view
using StatsBase: countmap

abstract type AbstractDataStructure end
abstract type AbstractObservation end

# ---------------------------------
# Simulation Draws
# ---------------------------------


const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

struct SimulationDraws{T, N, A<:AbstractRealArray{T,N}}
    u::A
    v::A
    psi1::A
    dpsidrho::A
    function SimulationDraws(u::A, v::A, psi1::A, dpsidrho::A) where {T,N,A<:AbstractRealArray{T,N}}
        size(dpsidrho) == size(psi1) == size(u) == size(v)  || throw(DimensionMismatch())
        return new{T,N,A}(u,v,psi1,dpsidrho)
    end
end

const SimulationDrawsVector{T,A} = SimulationDraws{T,1,A} where {T,A}
const SimulationDrawsMatrix{T,A} = SimulationDraws{T,2,A} where {T,A}


function SimulationDraws(M::Integer, N::Integer, bases::NTuple{2,Int}=(2,3); kwargs...)
    uvs = map((b) -> Matrix{Float64}(undef, M,N), bases)
    HaltonDraws!.(uvs, bases; kwargs...)
    psi1 = similar(uvs[1])
    dpsidrho = similar(uvs[1])
    return SimulationDraws(uvs..., psi1, dpsidrho)
end

_u(s::SimulationDraws) = s.u
_v(s::SimulationDraws) = s.v
_ψ1(s::SimulationDraws) = s.psi1
_ψ2(s::SimulationDraws) = _u(s)
_dψ1dρ(s::SimulationDraws) = s.dpsidrho
_psi1(s::SimulationDraws) = _ψ1(s)
_psi2(s::SimulationDraws) = _ψ2(s)

_dψdρ(s::SimulationDraws) = s.dpsidrho
@deprecate _dψdρ(s) _dψ1dρ(s)

function update_ψ1!(s::SimulationDraws, ρ::Real)
    # 0 <= ρ <= 1 || throw(DomainError())
    wt = sqrt(1-ρ^2)
    _ψ1(s) .= ρ.*_u(s)  +  wt.*_v(s)
end

function update_dψ1dρ!(s::SimulationDraws, ρ::Real)
    0 <= ρ <= 1 || throw(DomainError())
    rhoinvwt = ρ/sqrt(1-ρ^2)
    _dψ1dρ(s) .= _u(s) .- rhoinvwt .* _v(s)
end


# unobservables we integrate out
size(s::SimulationDraws) = size(_u(s))

tup(s::SimulationDraws) = _u(s), _v(s), _ψ1(s), _dψdρ(s)

view(s::SimulationDrawsMatrix, i::Integer) = SimulationDraws(view.(tup(s), :, i)...)
getindex(s::SimulationDrawsVector, m) = getindex.(tup(s), m)

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
@deprecate _choice(d) _y(d)


num_x(d::Union{ObservationRoyalty,DataRoyalty}) = size(_x(d), 1)
length(d::DataRoyalty) = length(_y(d))
size(d::DataRoyalty) = length(d)

function update_xbeta!(d::DataRoyalty, theta::AbstractVector)
    length(theta) == num_x(d) || throw(DimensionMismatch("theta: $(size(theta)) and x: $(size(_x(d)))"))
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
