# data strucutres
#---------------------------

struct SimulationDraws{T, N, A<:AbstractRealArray{T,N}}
    u::A
    v::A
    psi1::A
    dpsidrho::A
    qm::Vector{T}
    function SimulationDraws(u::A, v::A, psi1::A, dpsidrho::A, qm::Vector{T}) where {T,N,A<:AbstractRealArray{T,N}}
        size(dpsidrho) == size(psi1) == size(u) == size(v)  || throw(DimensionMismatch())
        length(qm) == size(u,1) || throw(DimensionMismatch())
        return new{T,N,A}(u,v,psi1,dpsidrho,qm)
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
    return SimulationDraws(uvs..., psi1, dpsidrho, qm)
end

# interface
#---------------------------

# access fields
_u(    s::SimulationDraws) = s.u
_v(    s::SimulationDraws) = s.v
_ψ1(   s::SimulationDraws) = s.psi1
_ψ2(   s::SimulationDraws) = _u(s)
_dψ1dρ(s::SimulationDraws) = s.dpsidrho
_psi1( s::SimulationDraws) = _ψ1(s)
_psi2( s::SimulationDraws) = _ψ2(s)
_qm(   s::SimulationDraws) = s.qm
_num_sim(s::SimulationDraws) = size(_u(s),1)

# manipulate like array
tup(s::SimulationDrawsMatrix) = _u(s), _v(s), _ψ1(s), _dψ1dρ(s)
tup(s::SimulationDrawsVector) = _u(s), _v(s), _ψ1(s), _dψ1dρ(s), _qm(s)

size(    s::SimulationDraws) = size(_u(s))
view(    s::SimulationDrawsMatrix, i::Integer) = SimulationDraws(view.(tup(s), :, i)..., _qm(s))
getindex(s::SimulationDrawsVector, m) = getindex.(tup(s), m)

# doing mean(psi) and mean(psi^2)
function psi2_wtd_sum_and_sumsq(s::SimulationDrawsVector{T}) where {T}
    qm = _qm(s)
    ψ = _ψ2(s)
    @assert sum(qm) ≈ 1
    ψbar = dot(qm,ψ) # zero(T)
    ψ2bar = sumprod3(qm,qm,ψ)
    # ψ2bar = zero(T)
    # @inbounds @simd for i = OneTo(length(qm))
    #     ψ2bar += ψ[i] * ( ψbar += ψ[i] * qm[i])
    # end
    return ψbar, ψ2bar
end




# updating
function zero!(s::SimulationDraws{T}) where {T}
    fill!(_qm(s), zero(T))
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
