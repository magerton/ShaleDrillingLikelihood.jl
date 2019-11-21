# abstract type AbstractDrillModel <: AbstractModel end  # from ShaleDrillingLikelihood.jl

"Static discrete choice model to test likelihood"
struct TestDrillModel <: AbstractDrillModel end
reward(x::TestDrillModel) = x

"Full-blown Dynamic discrete choice model"
struct DynamicDrillingModel{T<:Real, APF<:AbstractPayoffFunction, AUP<:AbstractUnitProblem, TT<:Tuple, AM<:AbstractMatrix{T}, AR<:StepRangeLen{T}}
    reward::APF           # payoff function
    discount::T           # discount factor
    statespace::AUP       # structure of endogenous choice vars
    zspace::TT            # z-space (tuple)
    ztransition::AM       # transition for z
    shockspace::AR        # ψspace = u, ρu + sqrt(1-ρ²)*v
    anticipate_t1ev::Bool # do we anticipate the ϵ shocks assoc w/ each choice?

    Emax::Array{T,3}
    dEmax::Array{T,4}

    tmpvars::DCDPTmpVars{T, SparseMatrixCSC{T,Int}, Array{T,3}, Array{T,4}}
end

# deprecate this!
const dcdp_primitives = DynamicDrillingModel

# struct dcdp_primitives{FF<:AbstractPayoffFunction,T<:Real,UP<:AbstractUnitProblem,AM<:AbstractMatrix{T},TT<:Tuple,AV<:AbstractVector{T}}
#     f::FF
#     β::T
#     wp::UP            # structure of endogenous choice vars
#     zspace::TT        # z-space (tuple)
#     Πz::AM            # transition for z
#     ψspace::AV        # ψspace = u + σv*v
#     anticipate_e::Bool # do we anticipate the ϵ shocks assoc w/ each choice?
# end

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
shockspace(     x::DynamicDrillingModel) = x.shockspace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev

@deprecate flow(x::DynamicDrillingModel)          reward(x)
@deprecate β(x::DynamicDrillingModel)             discount(x)
@deprecate wp(x::DynamicDrillingModel)            statespace(x)
@deprecate _zspace(x::DynamicDrillingModel)       zspace(x)
@deprecate Πz(x::DynamicDrillingModel)            ztransition(x)
@deprecate _ψspace(x::DynamicDrillingModel)       shockspace(x)
@deprecate anticipate_e(x::DynamicDrillingModel)  anticipate_t1ev(x)

struct dcdp_Emax
    EV::Array{Float64,3}
    dEV::Array{Float64,4}
end

struct DCDPTmpVars{T<:Real, AA3<:AbstractArray3{T}, AA4<:AbstractArray4{T}, SM<:AbstractMatrix{T}}
    ubVfull::AA3
    dubVfull::AA4
    q::AA3

    lse::Matrix{T}
    tmp::Matrix{T}
    tmp_cart::Matrix{CartesianIndex{3}}
    Πψtmp::Matrix{T}

    IminusTEVp::SM
end

const dcdp_tmpvars = DCDPTmpVars

# dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEVσ::AbstractArray3{T}) where {T} =  dcdp_Emax{T,typeof(EV),typeof(dEV)}(EV,dEV,dEVσ)
#
# function dcdp_Emax(p::dcdp_primitives{FF,T}) where {FF,T}
#     EV   = zeros(T, _nz(p), _nψ(p),          _nS(p))
#     dEV  = zeros(T, _nz(p), _nψ(p), _nθt(p), _nS(p))
#     dEVσ = zeros(T, _nz(p), _nψ(p),          _nSexp(p))
#     dcdp_Emax(EV,dEV,dEVσ) # ,dEVψ)
# end
