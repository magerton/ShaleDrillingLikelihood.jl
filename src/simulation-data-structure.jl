# data strucutres
#---------------------------

struct SimulationDraws{T, N, A<:AbstractRealArray{T,N}}
    u::A
    v::A
    psi1::A
    dpsidrho::A
    qm::Vector{T} # holds log L_{im}
    am::Vector{T} # pdf(η₁) / cdf(η₁)
    bm::Vector{T} # pdf(η₂) / cdf(η₂)
    cm::Vector{T} # (pdf(η₂) - pdf(η₁)) / (cdf(η₂) - cdf(η₁))
    function SimulationDraws(
        u::A, v::A, psi1::A, dpsidrho::A, qm::V, am::V, bm::V, cm::V
    ) where {T,N,A<:AbstractRealArray{T,N},V<:Vector{T}}
        size(dpsidrho) == size(psi1) == size(u) == size(v)  || throw(DimensionMismatch())
        length(qm) == length(am) == length(bm) == length(cm) == size(u,1) || throw(DimensionMismatch())
        return new{T,N,A}(u,v,psi1,dpsidrho,qm,am,bm,cm)
    end
end

const SimulationDrawsVector{T,A} = SimulationDraws{T,1,A} where {T,A}
const SimulationDrawsMatrix{T,A} = SimulationDraws{T,2,A} where {T,A}

# generate simulations using Halton
#---------------------------

function SimulationDraws(M::Integer, N::Integer, bases::NTuple{2,Int}=(2,3); kwargs...)
    uvs = map((b) -> Matrix{Float64}(undef, M,N), bases)
    HaltonDraws!.(uvs, bases; kwargs...)
    psi1 = similar(uvs[1])
    dpsidrho = similar(uvs[1])
    qm = Vector{Float64}(undef, M)
    am = similar(qm)
    bm = similar(qm)
    cm = similar(qm)
    return SimulationDraws(uvs..., psi1, dpsidrho, qm, am, bm, cm)
end

# interface
#---------------------------

# access fields
_u(    s::SimulationDraws) = s.u
_v(    s::SimulationDraws) = s.v
_ψ1(   s::SimulationDraws) = s.psi1
_dψ1dρ(s::SimulationDraws) = s.dpsidrho
_qm(   s::SimulationDraws) = s.qm
_am(   s::SimulationDraws) = s.am
_bm(   s::SimulationDraws) = s.bm
_cm(   s::SimulationDraws) = s.cm

_num_sim(s::SimulationDraws) = size(_u(s),1)
_ψ2(   s::SimulationDraws) = _u(s)
_psi1( s::SimulationDraws) = _ψ1(s)
_psi2( s::SimulationDraws) = _ψ2(s)
_llm(  s::SimulationDraws) = _qm(s)
_LLm(  s::SimulationDraws) = _llm(s)

# manipulate like array
tup(s::SimulationDrawsMatrix) = _u(s), _v(s), _ψ1(s), _dψ1dρ(s)
tup(s::SimulationDrawsVector) = _u(s), _v(s), _ψ1(s), _dψ1dρ(s), _qm(s)

size(    s::SimulationDraws) = size(_u(s))
view(    s::SimulationDrawsMatrix, i::Integer) = SimulationDraws(view.(tup(s), :, i)..., _qm(s), _am(s), _bm(s), _cm(s))
getindex(s::SimulationDrawsVector, m) = getindex.(tup(s), m)

# doing mean(psi) and mean(psi^2)
function psi2_wtd_sum_and_sumsq(s::SimulationDrawsVector{T}) where {T}
    qm = _qm(s)
    ψ = _ψ2(s)
    ψbar = dot(qm,ψ)
    ψ2bar = sumprod3(qm,ψ,ψ)
    return ψbar, ψ2bar
end


# so we can pre-calculate shocks
function update_ψ1!(s::SimulationDraws, ρ::Real)
    0 <= ρ <= 1 || @warn "0 <= ρ <= 1 is false"
    wt = sqrt(1-ρ^2)
    _ψ1(s) .= ρ.*_u(s) .+ wt.*_v(s)
    return nothing
end

function update_dψ1dρ!(s::SimulationDraws, ρ::Real)
    0 <= ρ <= 1 || @warn "0 <= ρ <= 1 is false"
    rhoinvwt = ρ/sqrt(1-ρ^2)
    _dψ1dρ(s) .= _u(s) .- rhoinvwt .* _v(s)
    return nothing
end
