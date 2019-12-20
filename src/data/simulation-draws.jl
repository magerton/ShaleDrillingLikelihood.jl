export SimulationDraws

# functions to define simulations

# convert θρ ∈ R ↦ ρ ∈ (0,1)
@inline _ρ(θρ) = logistic(θρ)
@inline _dρdθρ(θρ) = (z = logistic(θρ); z*(1-z) )
@inline _ρsq(θρ) = _ρ(θρ)^2
@deprecate _dρdσ(θρ) _dρdθρ(θρ)
# @deprecate _ρ2(θρ) _ρsq(θρ)

# go from iid normal u,v to correlated shocks
@inline _ψ1(u::Number, v::Number, ρ::Number) = ρ*u + sqrt(1-ρ^2)*v
@inline _ψ2(u::Number, v::Number, ρ::Number) = _ψ2(u,v)
@inline _ψ2(u::Number, v::Number) = u

# derivative of ψ1 wrt ρ and θρ
@inline _dψ1dρ( u::Number, v::Number, ρ::Number) = u - ρ/sqrt(1-ρ^2)*v
@inline _dψ1dθρ(u::Number, v::Number, ρ::Number, θρ::Number) = _dψ1dρ(u,v,ρ)*_dρdθρ(θρ)

# z(ψ2|ψ1)
@inline cond_z(x2::Number, x1::Number, Δ::Number, ρ::Number) = (x2 - ρ*x1 + Δ)/sqrt(1-ρ^2)
# @deprecate  _z(x2::Number, x1::Number, Δ::Number, ρ::Number) conditional_z(x2, x1, Δ, ρ)

# derivatives
@inline dcond_zdρ(x2::Number, x1::Number, ρ::Number, z::Number) = -x1/sqrt(1-ρ^2) + ρ*z/(1-ρ^2)
# @deprecate  _dzdρ(x2::Number, x1::Number, ρ::Number, z::Number) dcond_zdρ(x2,x1,ρ,z)

@inline _dcond_probdρ(x2::Number, x1::Number, Δ::Number, ρ::Number) = (z = cond_z(x2,x1,Δ,ρ);  normpdf(z) * dcond_zdρ(x2,x1,ρ,z))
# @deprecate      _dπdρ(x2::Number, x1::Number, Δ::Number, ρ::Number) _dcond_probdρ(x2,x1,Δ,ρ)

# finite difference versions
@inline _ρfd(x::Number, h::Number) = _ρ(x+h)
@inline cond_zfd(x2::Number, x1::Number, Δ::Number, ρ::Number, h::Number) = cond_z(x2, x1+h, Δ, ρ)
# @deprecate    _z(x2::Number, x1::Number, Δ::Number, ρ::Number, h::Number) cond_zfd(x2,x1,Δ,ρ,h)

# data strucutres
#---------------------------

struct SimulationDraw{T,V<:AbstractVector{T}}
    psi1::T
    u::T
    dpsidrho::T
    drillgradm::V
end

eltype(::SimulationDraw{T}) where {T} = T

function SimulationDraw(u::Real, v::Real, θρ::T) where {T}
    ρ = _ρ(θρ)
    psi1 = _ψ1(u,v,ρ)
    @assert u == _ψ2(u,v,ρ)
    dpsidrho = _dψ1dθρ(u,v,ρ,θρ)
    zerovec = Vector{T}(undef,0)
    return SimulationDraw(psi1, u, dpsidrho, zerovec)
end

function SimulationDrawFD(s::SimulationDraw, x::Real)
    SimulationDraw(_ψ1(s)+x, _u(s)+x, _dψ1dθρ(s), _drillgradm(s))
end

SimulationDraw(θρ) = SimulationDraw(randn(), randn(), θρ)

struct SimulationDraws{T, N, A<:AbstractRealArray{T,N}}
    u::A
    v::A
    psi1::A
    dpsidrho::A
    qm::Vector{T} # holds log L_{im}
    am::Vector{T} # pdf(η₁) / cdf(η₁)
    bm::Vector{T} # pdf(η₂) / cdf(η₂)
    cm::Vector{T} # (pdf(η₂) - pdf(η₁)) / (cdf(η₂) - cdf(η₁))
    drillgradm::Matrix{T}

    function SimulationDraws(
        u::A, v::A, psi1::A, dpsidrho::A, qm::V, am::V, bm::V, cm::V,
        drillgradm::Matrix{T}
    ) where {T,N,A<:AbstractRealArray{T,N},V<:Vector{T}}
        size(dpsidrho) == size(psi1) == size(u) == size(v)  || throw(DimensionMismatch())
        length(qm) == length(am) == length(bm) == length(cm) == size(u,1) || throw(DimensionMismatch())
        length(qm) == size(drillgradm,2) || throw(DimensionMismatch())
        return new{T,N,A}(u,v,psi1,dpsidrho,qm,am,bm,cm,drillgradm)
    end
end

const SimulationDrawsVector{T,A} = SimulationDraws{T,1,A} where {T,A}
const SimulationDrawsMatrix{T,A} = SimulationDraws{T,2,A} where {T,A}
const SimulationDrawOrDraws = Union{<:SimulationDraws,<:SimulationDraw}

# generate simulations using Halton
#---------------------------

function SimulationDraws(M::Integer, N::Integer, K::Integer=0; bases::NTuple{2,Int}=(2,3))
    uvs = map((b) -> Matrix{Float64}(undef, M,N), bases)
    HaltonDraws!.(uvs, bases; skip=5000, distr=Normal())
    psi1 = similar(uvs[1])
    dpsidrho = similar(uvs[1])
    qm = Vector{Float64}(undef, M)
    am = similar(qm)
    bm = similar(qm)
    cm = similar(qm)
    drillgradm = Matrix{Float64}(undef,K,M)
    return SimulationDraws(uvs..., psi1, dpsidrho, qm, am, bm, cm, drillgradm)
end

function SimulationDraws(M, data::AbstractDataSet; kwargs...)
    return SimulationDraws(M, length(data), _nparm(_model(data)); kwargs...)
end

function SimulationDraws(M, data::AbstractDataSetofSets; kwargs...)
    drillmod = _model(drill(data))
    return SimulationDraws(M, num_i(data), _nparm(drillmod); kwargs...)
end


# interface
#---------------------------

# access fields
_v(    s::SimulationDraws) = s.v
_u(    s::SimulationDrawOrDraws) = s.u
_ψ1(   s::SimulationDrawOrDraws) = s.psi1
_dψ1dθρ(s::SimulationDrawOrDraws) = s.dpsidrho
_qm(   s::SimulationDrawOrDraws) = s.qm
_drillgradm(s::SimulationDrawOrDraws) = s.drillgradm
_am(   s::SimulationDraws) = s.am
_bm(   s::SimulationDraws) = s.bm
_cm(   s::SimulationDraws) = s.cm

_ψ2(   s::SimulationDrawOrDraws) = _u(s)
_psi1( s::SimulationDrawOrDraws) = _ψ1(s)
_psi2( s::SimulationDrawOrDraws) = _ψ2(s)
_llm(  s::SimulationDrawOrDraws) = _qm(s)

# @deprecate _LLm(s::SimulationDrawOrDraws) _llm(s)
# @deprecate _dψ1dρ(s::SimulationDrawOrDraws) _dψ1dθρ(s)

_num_sim(s::SimulationDraws) = size(_u(s),1)
num_i(s::SimulationDrawsMatrix) = size(_u(s),2)
num_i(s::SimulationDrawsVector) = 1

_num_sim(M::Integer) = M

# manipulate like array
tup(s::SimulationDraws) = _u(s), _v(s), _ψ1(s), _dψ1dθρ(s)
size(s::SimulationDraws) = size(_u(s))
_nparm(s::SimulationDraws) = size(_drillgradm(s),1)

function view(s::SimulationDrawsMatrix, i::Integer)
    psiviews = view.(tup(s), :, i)
    SimulationDraws(psiviews..., _qm(s), _am(s), _bm(s), _cm(s), _drillgradm(s))
end

function getindex(s::SimulationDrawsVector, m)
    u, v, ψ1, dψ1dρ = getindex.(tup(s), m)
    gradmvw = view(_drillgradm(s), :, m)
    return SimulationDraw(ψ1, u, dψ1dρ, gradmvw)
end

# doing mean(psi) and mean(psi^2)
function psi2_wtd_sum_and_sumsq(s::SimulationDrawsVector{T}) where {T}
    qm = _qm(s)
    ψ = _ψ2(s)
    ψbar = dot(qm,ψ)
    ψ2bar = sumprod3(qm,ψ,ψ)
    return ψbar, ψ2bar
end


# so we can pre-calculate shocks
function update_ψ1!(ψ1::AA, u::AA, v::AA, θρ::Real) where {AA<:AbstractArray}
    size(ψ1) == size(u) == size(v) || throw(DimensionMismatch())
    ρ = _ρ(θρ)
    0 <= ρ <= 1 || @warn "0 <= ρ=$ρ <= 1 is false. Have θρ = $θρ"
    ψ1 .= _ψ1.(u, v, ρ)
    return nothing
end


function update_dψ1dθρ!(dψ1dθρ::A, u::A, v::A, θρ::Real) where {A<:AbstractArray}
    size(dψ1dθρ) == size(u) == size(v) || throw(DimensionMismatch())
    ρ = _ρ(θρ)
    0 <= ρ <= 1 || @warn "0 <= ρ=$ρ <= 1 is false. Have θρ = $θρ"
    dψ1dθρ .= _dψ1dθρ.(u,v,ρ,θρ)
    return nothing
end

update_ψ1!(    s::SimulationDraws, θρ) = update_ψ1!(     _ψ1(s),    _u(s), _v(s), θρ)
update_dψ1dθρ!(s::SimulationDraws, θρ) = update_dψ1dθρ!(_dψ1dθρ(s), _u(s), _v(s), θρ)

function update!(sim::SimulationDraws,θρ::Real)
    update_ψ1!(sim, θρ)
    update_dψ1dθρ!(sim, θρ)
end
