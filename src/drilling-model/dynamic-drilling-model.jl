export DynamicDrillingModel,
    reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev


"Full-blown Dynamic discrete choice model"
struct DynamicDrillingModel{T<:Real, PF<:DrillReward, AM<:AbstractMatrix{T}, AUP<:AbstractUnitProblem, TT<:Tuple, AR<:StepRangeLen{T}} <: AbstractDrillModel
    reward::PF            # payoff function
    discount::T           # discount factor
    statespace::AUP       # structure of endogenous choice vars
    zspace::TT            # z-space (tuple)
    ztransition::AM       # transition for z
    psispace::AR          # ψspace = (u, ρu + sqrt(1-ρ²)*v)
    anticipate_t1ev::Bool # do we anticipate the ϵ shocks assoc w/ each choice?

    function DynamicDrillingModel(reward::APF, discount::T, statespace::AUP, zspace::TT, ztransition::AM, psispace::AR, anticipate_t1ev) where {T,N, APF, AUP, TT<:NTuple{N,AbstractRange}, AM, AR}
        nz = checksquare(ztransition)
        npsi = length(psispace)
        nS = length(statespace)
        nd = length(actionspace(statespace))
        ntheta = _nparm(reward)

        0 < discount < 1 || throw(DomainError(discount))
        nz == prod(length.(zspace)) || throw(DimensionMismatch("zspace dim != ztransition dim"))

        return new{T,APF,AM,AUP,TT,AR}(reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev)
    end
end

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
psispace(       x::DynamicDrillingModel) = x.psispace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev

beta_1minusbeta(ddm::DynamicDrillingModel) = discount(ddm) / (1-discount(ddm))

# -----------------------------------------
# components of stuff
# -----------------------------------------

# @deprecate revenue(x::ObservationDrill) revenue(_model(x))
# @deprecate drill(  x::ObservationDrill) drill(  _model(x))
# @deprecate extend( x::ObservationDrill) extend( _model(x))
# @deprecate extensioncost(x::DrillModel) extend(x)
# @deprecate drillingcost( x::DrillModel) drill(x)
#
# @deprecate flow(x::DynamicDrillingModel)          reward(x)
# @deprecate β(x::DynamicDrillingModel)             discount(x)
# @deprecate wp(x::DynamicDrillingModel)            statespace(x)
# @deprecate _zspace(x::DynamicDrillingModel)       zspace(x)
# @deprecate Πz(x::DynamicDrillingModel)            ztransition(x)
# @deprecate _ψspace(x::DynamicDrillingModel)       psispace(x)
# @deprecate anticipate_e(x::DynamicDrillingModel)  anticipate_t1ev(x)



"Value function arrays for dynamic model"
struct DCDPEmax{T<:Real}
    EV::Array{T,3}
    dEV::Array{T,4}
    function DCDPEmax(EV, dEV)
         nz, npsi, nk, ns  = size(dEV)
        (nz, npsi,     ns) == size(EV)  || throw(DimensionMismatch("EV is $(size(EV)) but dEV is $(size(dEV))"))
        new{eltype(EV)}(EV, dEV)
    end
end

EV(x::DCDPEmax) = x.EV
dEV(x::DCDPEmax) = x.dEV

size(x::DCDPEmax) = size(dEV(x))

function fill!(x::DCDPEmax, y)
    fill!(EV(x), y)
    fill!(dEV(x), y)
end

const dcdp_Emax = DCDPEmax

function DCDPEmax(ddm::DynamicDrillingModel{T}) where {T}
    nz = size(ztransition(ddm), 1)
    nψ = length(psispace(ddm))
    nK = _nparm(reward(ddm))
    nS = length(statespace(ddm))

    dev = Array{T,4}(undef, nz, nψ, nK, nS)
     ev = Array{T,3}(undef, nz, nψ,     nS)
    return DCDPEmax(ev, dev)
end

# EmaxArrays::DCDPEmax{T}
# tmpvars::DCDPTmpVars{T, AM, Array{T,3}, Array{T,4}}
# size(Emax) == (nz, npsi, nS) || throw(DimensionMismatch())
# size(tmpv) == (ntheta nz, npsi, nd) || throw(DimensionMismatch())

""""
    ensure_diagonal(x)

Create (possibly sparse) matrix which ensures diagonal has all entries (even if zero).
"""
function ensure_diagonal(x)
    n = checksquare(x)
    A = x + SparseMatrixCSC(I, n, n)
    Arows = rowvals(A)
    Avals = nonzeros(A)
    for j = OneTo(n)
       for i in nzrange(A, j)
          j == Arows[i]  &&  (Avals[i] -= 1)
       end
    end
    return A
end


"Temp vars for dynamic model"
struct DCDPTmpVars{T<:Real, SM<:AbstractMatrix{T}, AA3<:AbstractArray3{T}, AA3b<:AbstractArray3{T}, AA4<:AbstractArray4{T}} <: AbstractTmpVars
    ubVfull::AA3
    dubVfull::AA4
    dubVfullperm::AA4
    q::AA3b
    lse::Matrix{T}
    tmp::Matrix{T}
    tmp_cart::Matrix{CartesianIndex{3}}
    Πψtmp::Matrix{T}
    IminusTEVp::SM

    function DCDPTmpVars(ubVfull::AA3, dubVfull::AA4, dubVfullperm::AA4, q::AA3b, lse, tmp, tmp_cart, Πψtmp, IminusTEVp::SM) where {AA3, AA3b, AA4, SM}
        (nθt, nz, nψ, nd) = size(dubVfull)
        (     nz, nψ, nθt, nd) == size(dubVfullperm) || throw(DimensionMismatch())
        (nz, nψ, nd) == size(ubVfull) == size(q) || throw(DimensionMismatch())
        (nz,nψ) == size(lse) == size(tmp) == size(tmp_cart) || throw(DimensionMismatch())
        nψ == checksquare(Πψtmp) || throw(DimensionMismatch())
        nz == checksquare(IminusTEVp) || throw(DimensionMismatch())
        T = eltype(ubVfull)
        return new{T,SM,AA3,AA3b,AA4}(ubVfull, dubVfull, dubVfullperm, q, lse, tmp, tmp_cart, Πψtmp, IminusTEVp)
    end
end

const dcdp_tmpvars = DCDPTmpVars

ubVfull(     x::DCDPTmpVars) = x.ubVfull
dubVfull(    x::DCDPTmpVars) = x.dubVfull
dubVfullperm(x::DCDPTmpVars) = x.dubVfullperm
q(           x::DCDPTmpVars) = x.q
lse(         x::DCDPTmpVars) = x.lse
tmp(         x::DCDPTmpVars) = x.tmp
tmp_cart(    x::DCDPTmpVars) = x.tmp_cart
Πψtmp(       x::DCDPTmpVars) = x.Πψtmp
IminusTEVp(  x::DCDPTmpVars) = x.IminusTEVp

size(x::DCDPTmpVars) = size(dubVfullperm(x))

ubV(     x::DCDPTmpVars{T,SM,AA3}) where {T,SM,AA3<:SubArray} = ubVfull(x)
dubV(    x::DCDPTmpVars{T,SM,AA3}) where {T,SM,AA3<:SubArray} = dubVfull(x)
dubVperm(x::DCDPTmpVars{T,SM,AA3}) where {T,SM,AA3<:SubArray} = dubVfullperm(x)

@deprecate _ubVfull(   x::DCDPTmpVars)   ubVfull(   x)
@deprecate _dubVfull(  x::DCDPTmpVars)   dubVfull(  x)
@deprecate _q(         x::DCDPTmpVars)   q(         x)
@deprecate _lse(       x::DCDPTmpVars)   lse(       x)
@deprecate _tmp(       x::DCDPTmpVars)   tmp(       x)
@deprecate _tmp_cart(  x::DCDPTmpVars)   tmp_cart(  x)
@deprecate _Πψtmp(     x::DCDPTmpVars)   Πψtmp(     x)
@deprecate _IminusTEVp(x::DCDPTmpVars)   IminusTEVp(x)
@deprecate _ubV(       x::DCDPTmpVars{T,SM,AA3}) where {T,SM,AA3} ubV(  x)
@deprecate _dubV(      x::DCDPTmpVars{T,SM,AA3}) where {T,SM,AA3} dubV( x)

size(x::DCDPTmpVars) = size(dubVfull(x))

function fill!(t::DCDPTmpVars, x)
    fill!(ubVfull(     t), x)
    fill!(dubVfull(    t), x)
    fill!(dubVfullperm(t), x)
    fill!(q(           t), x)
    fill!(lse(         t), x)
    fill!(tmp(         t), x)
    fill!(tmp_cart(    t), x)
    fill!(Πψtmp(       t), x)
end

function DCDPTmpVars(nθt, nz, nψ, nd, ztransition::AbstractMatrix{T}) where {T<:Real}
    dubVfull = Array{T,4}(undef, nθt, nz, nψ,      nd)
    dubVperm = Array{T,4}(undef,      nz, nψ, nθt, nd)
    ubVfull  = Array{T,3}(undef,      nz, nψ,      nd)
    q = similar(ubVfull)
    lse = Matrix{T}(undef, nz, nψ)
    tmp = similar(lse)
    tmp_cart = similar(lse, CartesianIndex{3})
    Πψtmp = Matrix{T}(undef, nψ, nψ)
    IminusTEVp = ensure_diagonal(ztransition)
    return DCDPTmpVars(ubVfull, dubVfull, dubVperm, q, lse, tmp, tmp_cart, Πψtmp, IminusTEVp)
end

function DCDPTmpVars(x::DynamicDrillingModel)
    piz = ztransition(x)
    nθt = _nparm(reward(x))
    nz = size(piz, 1)
    nψ = length(psispace(x))
    nd = num_choices(statespace(x))
    DCDPTmpVars(nθt, nz, nψ, nd, piz)
end

function view(t::DCDPTmpVars, idxd::AbstractVector)
    first(idxd) == 1 || throw(DomainError())
    last(idxd) <= size(ubVfull(t),3) || throw(DomainError())
    @views ubV  = view(ubVfull(t), :,:,  idxd)
    @views dubV = view(dubVfull(t),:,:,:,idxd)
    @views dubvperm = view(dubVfullperm(t),:,:,:,idxd)
    @views qq   = view(q(t), :,:,idxd)
    return dcdp_tmpvars(ubV, dubV, dubvperm, qq, lse(t), tmp(t), tmp_cart(t), Πψtmp(t), IminusTEVp(t))
end


# --------------------------------------------------------
# Fill reward matrices
# --------------------------------------------------------

function flow!(tmpv::DCDPTmpVars, ddm::DynamicDrillingModel, θ::AbstractVector, sidx::Integer, ichars::Tuple, dograd::Bool)
    ψspace = psispace(ddm)
    zs = product(zspace(ddm)...)
    zψpdct = product(zs, ψspace)
    nk = _nparm(reward(ddm))
    nc = num_choices(statespace(ddm), sidx)

    sztup = (length(zs), length(ψspace), nk, nc)
    size(tmpv) == sztup || throw(DimensionMismatch("size tmvp = $(size(tmpv)) vs $sztup"))

    dubv  = reshape(dubVfull(tmpv),    nk,  :, nc)
    dubvp = reshape(dubVfullperm(tmpv), :, nk, nc)
    ubv   = reshape( ubVfull(tmpv),     :, nc)

    @threads for dp1 in dp1space(statespace(ddm), sidx)
        @inbounds for (i, (z,ψ)) in enumerate(zψpdct)
            obs = ObservationDrill(ddm, ichars, z, dp1-1, sidx)
            grad = view(dubv, :, i, dp1)
            ubv[i,dp1] = flow!(grad, obs, θ, ψ, dograd)
        end
    end
    permutedims!(dubvp, dubv, [2,1,3])
end
