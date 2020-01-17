using LoopVectorization
using Base: OneTo
using LinearAlgebra: stride1
using Test

add_1_dim(x::AbstractArray) = reshape(x, size(x)..., 1)
check_finite(x::AbstractArray) = all(isfinite.(x)) || throw(error("x not finite!"))

"Given k-dimensional array `x` where `n=size(x,k)`, compute multinomial logistic Pr(i ∈ 1:n | x[:, ..., :, k] )"
function softmax3!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
    ndims(q) == 1+ndims(lse) || throw(DimensionMismatch())
    xsizes = size(x)
    xsizes == size(q) || throw(DimensionMismatch("size(x) = $(size(x)) but size(q) = $(size(q))"))
    nk = last(xsizes)
    for i = OneTo(ndims(lse))
        size(q,i) == size(lse,i) == size(tmpmax,i) || throw(DimensionMismatch("size(x) = $(size(x)),  size(lse) = $(size(lse)), and size(tmpmax) = $(size(tmpmax))"))
    end
    0 < maxk <= nk || throw(DomainError(maxk))
    1 == stride1(q) == stride1(x) || throw(error("Arrays not strided"))

    isempty(x) && throw(error("x empty"))
    check_finite(x)

    maximum!(add_1_dim(tmpmax), x)
    fill!(lse, zero(T))

    xx = reshape(x, :, nk)
    qq = reshape(q, :, nk)

    for k in OneTo(nk)
        for i in eachindex(lse)
            tmp = exp(xx[i,k] - tmpmax[i])
            lse[i] += tmp
            k <= maxk && (qq[i,k] = tmp)
        end
    end

    qq[:,OneTo(maxk)] ./= vec(lse)
end

function softmax3_avx_broken!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        ndims(q) == 1+ndims(lse) || throw(DimensionMismatch())
        xsizes = size(x)
        xsizes == size(q) || throw(DimensionMismatch("size(x) = $(size(x)) but size(q) = $(size(q))"))
        nk = last(xsizes)
        for i = OneTo(ndims(lse))
            size(q,i) == size(lse,i) == size(tmpmax,i) || throw(DimensionMismatch("size(x) = $(size(x)),  size(lse) = $(size(lse)), and size(tmpmax) = $(size(tmpmax))"))
        end
        0 < maxk <= nk || throw(DomainError(maxk))
        1 == stride1(q) == stride1(x) || throw(error("Arrays not strided"))

        isempty(x) && throw(error("x empty"))
        check_finite(x)

        maximum!(add_1_dim(tmpmax), x)
        fill!(lse, zero(T))

        xx = reshape(x, :, nk)
        qq = reshape(q, :, nk)

        for k in OneTo(maxk)
            @avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                # FIXME - would prefer to replace 2nd loop w/ if stmt
                # k <= maxk && (qq[i,k] = tmp)
                qq[i,k] = tmp
            end
        end

        for k in maxk+1:nk
            @avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
            end
        end

        qq[:,OneTo(maxk)] ./= vec(lse)
end

function softmax3_avx_fixed!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        ndims(q) == 1+ndims(lse) || throw(DimensionMismatch())
        xsizes = size(x)
        xsizes == size(q) || throw(DimensionMismatch("size(x) = $(size(x)) but size(q) = $(size(q))"))
        nk = last(xsizes)
        for i = OneTo(ndims(lse))
            size(q,i) == size(lse,i) == size(tmpmax,i) || throw(DimensionMismatch("size(x) = $(size(x)),  size(lse) = $(size(lse)), and size(tmpmax) = $(size(tmpmax))"))
        end
        0 < maxk <= nk || throw(DomainError(maxk))
        1 == stride1(q) == stride1(x) || throw(error("Arrays not strided"))

        isempty(x) && throw(error("x empty"))
        check_finite(x)

        maximum!(add_1_dim(tmpmax), x)
        fill!(lse, zero(T))

        xx = reshape(x, :, nk)
        qq = reshape(q, :, nk)
        tmpmaxvec = vec(tmpmax)  # Always required??
        lsevec = vec(lse)        # Always required??

        for k in OneTo(maxk)
            qk = view(qq, :, k) # required if using a View
            xk = view(xx, :, k) # required if using a View
            @avx for i in eachindex(lsevec)
                tmp = exp(xk[i] - tmpmaxvec[i])
                lsevec[i] += tmp
                qk[i] = tmp
            end
        end

        for k in maxk+1:nk
            qk = view(qq, :, k)
            xk = view(xx, :, k)
            @avx for i in eachindex(lsevec)
                tmp = exp(xk[i] - tmpmaxvec[i])
                lsevec[i] += tmp
            end
        end

        qq[:,OneTo(maxk)] ./= vec(lse)
end


ni, nj, nk = (100, 100, 10)
x = rand(ni, nj, nk)
q = similar(x)
tmpmax = zeros(ni,nj)
lse = similar(tmpmax)

for f! in (softmax3!, softmax3_avx_fixed!, softmax3_avx_broken!)
    @testset "$(f!) with arrays" begin
        fill!(q,0)
        fill!(lse,0)
        @show @code_warntype f!(q, lse, tmpmax, x, 1)
        f!(q, lse, tmpmax, x, 1)
        @test all(sum(q; dims=3) .<= 1)
        fill!(q,0)
        fill!(lse,0)
        f!(q, lse, tmpmax, x)
        @test sum(q; dims=3) ≈ ones(ni,nj)
    end

    @testset "$(f!) with views" begin
        nkm1 = nk-1
        @views qvw = q[:,:,1:nkm1]
        @views xvw = x[:,:,1:nkm1]
        fill!(q,0)
        fill!(lse,0)
        @show @code_warntype f!(qvw, lse, tmpmax, xvw, 1)
        f!(qvw, lse, tmpmax, xvw, 1)
        @test all(sum(qvw; dims=3) .<= 1)
        fill!(q,0)
        fill!(lse,0)
        f!(qvw, lse, tmpmax, xvw)
        @test sum(qvw; dims=3) ≈ ones(ni,nj)
    end
end
