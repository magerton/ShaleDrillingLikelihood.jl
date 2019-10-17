# functions to define simulations

# convert θρ ∈ R ↦ ρ ∈ (0,1)
@inline _ρ(θρ) = θρ # logistic(θρ)
@inline _dρdθρ(θρ) = one(eltype(θρ)) # = (z = logistic(θρ); z*(1-z) )
@inline _ρsq(θρ) = _ρ(θρ)^2
@deprecate _dρdσ(θρ) _dρdθρ(θρ)
@deprecate _ρ2(θρ) _ρsq(θρ)

# go from iid normal u,v to correlated shocks
@inline _ψ1(u::Number, v::Number, ρ::Number) = ρ*u + sqrt(1-ρ^2)*v
@inline _ψ2(u::Number, v::Number, ρ::Number) = _ψ2(u,v)
@inline _ψ2(u::Number, v::Number) = u

# derivative of ψ1 wrt ρ and θρ
@inline _dψ1dρ( u::Number, v::Number, ρ::Number) = u - ρ/sqrt(1-ρ^2)*v
@inline _dψ1dθρ(u::Number, v::Number, ρ::Number, θρ::Number) = _dψ1dρ(u,v,ρ)*_dρdθρ(θρ)

# z(ψ2|ψ1)
@inline cond_z(x2::Number, x1::Number, Δ::Number, ρ::Number) = (x2 - ρ*x1 + Δ)/sqrt(1-ρ^2)
@deprecate  _z(x2::Number, x1::Number, Δ::Number, ρ::Number) conditional_z(x2, x1, Δ, ρ)

# derivatives
@inline dcond_zdρ(x2::Number, x1::Number, ρ::Number, z::Number) = -x1/sqrt(1-ρ^2) + ρ*z/(1-ρ^2)
@deprecate  _dzdρ(x2::Number, x1::Number, ρ::Number, z::Number) dcond_zdρ(x2,x1,ρ,z)

@inline _dcond_probdρ(x2::Number, x1::Number, Δ::Number, ρ::Number) = (z = cond_z(x2,x1,Δ,ρ);  normpdf(z) * dcond_zdρ(x2,x1,ρ,z))
@deprecate      _dπdρ(x2::Number, x1::Number, Δ::Number, ρ::Number) _dcond_probdρ(x2,x1,Δ,ρ)

# finite difference versions
@inline _ρfd(x::Number, h::Number) = _ρ(x+h)
@inline cond_zfd(x2::Number, x1::Number, Δ::Number, ρ::Number, h::Number) = cond_z(x2, x1+h, Δ, ρ)
@deprecate    _z(x2::Number, x1::Number, Δ::Number, ρ::Number, h::Number) cond_zfd(x2,x1,Δ,ρ,h)

# data strucutres
#---------------------------

struct SimulationDraw{T}
    psi1::T
    u::T
    dpsidrho::T
    qm::T
end

function SimulationDraw(θρ::Float64)
    ρ = _ρ(θρ)
    u,v = randn(2)
    return SimulationDraw(_ψ1(u,v,ρ), _ψ2(u,v,ρ), _dψ1dθρ(u,v,ρ,θρ), 1.0)
end

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
const SimulationDrawOrDraws = Union{<:SimulationDraws,<:SimulationDraw}

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
_v(    s::SimulationDraws) = s.v
_u(    s::SimulationDrawOrDraws) = s.u
_ψ1(   s::SimulationDrawOrDraws) = s.psi1
_dψ1dρ(s::SimulationDrawOrDraws) = s.dpsidrho
_qm(   s::SimulationDrawOrDraws) = s.qm
_am(   s::SimulationDraws) = s.am
_bm(   s::SimulationDraws) = s.bm
_cm(   s::SimulationDraws) = s.cm

_ψ2(   s::SimulationDrawOrDraws) = _u(s)
_psi1( s::SimulationDrawOrDraws) = _ψ1(s)
_psi2( s::SimulationDrawOrDraws) = _ψ2(s)
_llm(  s::SimulationDrawOrDraws) = _qm(s)

@deprecate _LLm(s::SimulationDrawOrDraws) _llm(s)

_num_sim(s::SimulationDraws) = size(_u(s),1)

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
function update_ψ1!(ψ1::AA, u::AA, v::AA, ρ::Real) where {AA<:AbstractArray}
    size(ψ1) == size(u) == size(v) || throw(DimensionMismatch())
    0 <= ρ <= 1 || @warn "0 <= ρ <= 1 is false"
    ψ1 .= _ψ1.(u, v, ρ)
    return nothing
end


function update_dψ1dρ!(dψ1dρ::A, u::A, v::A, ρ::Real) where {A<:AbstractArray}
    size(dψ1dρ) == size(u) == size(v) || throw(DimensionMismatch())
    0 <= ρ <= 1 || @warn "0 <= ρ <= 1 is false"
    dψ1dρ .= _dψ1dρ.(u,v,ρ)
    return nothing
end

update_ψ1!(s::SimulationDraws, ρ)    = update_ψ1!(   _ψ1(s),    _u(s), _v(s), ρ)
update_dψ1dρ!(s::SimulationDraws, ρ) = update_dψ1dρ!(_dψ1dρ(s), _u(s), _v(s), ρ)

function update!(sim::SimulationDraws,ρ)
    update_ψ1!(sim, ρ)
    update_dψ1dρ!(sim, ρ)
end
