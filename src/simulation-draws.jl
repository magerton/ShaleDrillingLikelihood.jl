import Base: getindex, view, size

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
