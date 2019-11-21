"Value function arrays for dynamic model"
struct DCDPEmax{T<:Real}
    EV::Array{T,3}
    dEV::Array{T,4}
    function DCDPEmax(EV, dEV)
        Base.tail(size(dEV)) == size(EV) || throw(DimensionMismatch())
        new{eltype(EV)}(EV, dEV)
    end
end

EV(x::DCDPEmax) = x.EV
dEV(x::DCDPEmax) = x.dEV

size(x::DCDPEmax) = size(dEV(x))

function fill!(x, y)
    fill!(EV(x), y)
    fill!(dEV(x), y)
end

const dcdp_Emax = DCDPEmax

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
struct DCDPTmpVars{T<:Real, AA3<:AbstractArray3{T}, AA4<:AbstractArray4{T}, SM<:AbstractMatrix{T}} <: AbstractTmpVars
    ubVfull::AA3
    dubVfull::AA4
    q::AA3
    lse::Matrix{T}
    tmp::Matrix{T}
    tmp_cart::Matrix{CartesianIndex{3}}
    Πψtmp::Matrix{T}
    IminusTEVp::SM

    function DCDPTmpVars(ubVfull::AA3{T}, dubVfull::AA4, q::AA3, lse, tmp, tmp_cart, Πψtmp, IminusTEVp::SM) where {T,AA3, AA4, SM}
        (nθt, nz, nψ, nd) = size(dubVfull)
        (nz, nψ, nd) == size(ubVfull) == size(q) || throw(DimensionMismatch())
        (nz,nψ) == size(lse) == size(tmp) == size(tmp_cart) || throw(DimensionMismatch())
        nψ == checksquare(Πψtmp) || throw(DimensionMismatch())
        nz == checksquare(IminusTEVp) || throw(DimensionMismatch())
        return new{T,AA3,AA4,SM}(ubVfull, dubVfull, q, lse, tmp, tmp_cart, Πψtmp, IminusTEVp)
    end
end

const dcdp_tmpvars = DCDPTmpVars

_ubVfull(   x::DCDPTmpVars) = x.ubVfull
_dubVfull(  x::DCDPTmpVars) = x.dubVfull
_q(         x::DCDPTmpVars) = x.q
_lse(       x::DCDPTmpVars) = x.lse
_tmp(       x::DCDPTmpVars) = x.tmp
_tmp_cart(  x::DCDPTmpVars) = x.tmp_cart
_Πψtmp(     x::DCDPTmpVars) = x.Πψtmp
_IminusTEVp(x::DCDPTmpVars) = x.IminusTEVp

size(x::DCDPTmpVars) = size(_dubVfull(x))

function fill!(t::DCDPTmpVars, x)
    fill!(_ubVfull(   t), x)
    fill!(_dubVfull(  t), x)
    fill!(_q(         t), x)
    fill!(_lse(       t), x)
    fill!(_tmp(       t), x)
    fill!(_tmp_cart(  t), x)
    fill!(_Πψtmp(     t), x)
end

function DCDPTmpVars(nθt, nz, nψ, nd, ztransition::AbstractMatrix{T}) where {T<:Real}
    dubVfull = Array{T,4}(undef, nθt, nz, nψ, nd)
    ubVfull  = Array{T,3}(undef, nz, nψ, nd)
    q = similar(ubVfull)
    lse = Matrix{T}(undef, nz, nψ)
    tmp = similar(lse)
    tmp_cart = similar(lse, CartesianIndex{3})
    Πψtmp = Matrix{T}(nψ, nψ)
    IminusTEVp = ensure_diagonal(ztransition)
    return DCDPTmpVars(ubVfull, dubVfull, q, lse, tmp, tmp_cart, Πψtmp, IminusTEVp)
end


function view(t::DCDPTmpVars, idxd::AbstractVector)
    first(idxd) == 1 || throw(DomainError())
    last(idxd) <= size(_ubVfull(t),3) || throw(DomainError())
    @views ubV  = view(_ubVfull(t), :,:,  idxd)
    @views dubV = view(_dubVfull(t),:,:,:,idxd)
    @views q    = view(_q(t), :,:,idxd)
    return dcdp_tmpvars(ubV, dubV, q, _lse(t), _tmp(t), _tmp_cart(t), _Πψtmp(t), _IminusTEVp(t))
end
