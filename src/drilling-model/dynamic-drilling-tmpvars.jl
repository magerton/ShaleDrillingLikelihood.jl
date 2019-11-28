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
    check_rowvals_equal(A,x) || throw(error("rowvals are not the same..."))
    return A
end

check_rowvals_equal(A::SparseMatrixCSC, X::SparseMatrixCSC) = rowvals(A) == rowvals(X)
check_rowvals_equal(A, X) = true


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

function update_static_payoffs!(tmpv::DCDPTmpVars, ddm::DynamicDrillingModel, θ::AbstractVector, sidx::Integer, ichars::Tuple, dograd::Bool)
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

@deprecate flow!(t::DCDPTmpVars, ddm::DynamicDrillingModel, θ::AbstractVector, sidx::Integer, ichars::Tuple, dograd::Bool) update_static_payoffs!(t,ddm,θ,sidx,ichars,dograd)
