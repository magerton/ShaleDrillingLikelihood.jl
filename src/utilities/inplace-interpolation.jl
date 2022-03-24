# this set of functions sets up a pre-allocated set of arrays for interpolations
# we do this so that we can pre-allocate everything for value functions
# and be able to do interpolation stuff without creating new arrays

# padded coefs
function padded_coefs(::Type{TC}, A::AbstractArray, it) where {TC}
    indsA = axes(A)
    indspad = Interpolations.padded_axes(indsA, it)
    return Interpolations.padded_similar(TC, indspad)
end
padded_coefs(A,it) = padded_coefs(Interpolations.tcoef(A), A, it)

# padded coefs
struct InPlaceInterp{T, TC, N, A<:AbstractArray{T,N}, TCoefs<:AbstractArray{TC,N}, IT<:Interpolations.DimSpec{BSpline}, RT<:NTuple{N,AbstractRange},ITP<:Interpolations.BSplineInterpolation,SITP<:ScaledInterpolation}
    x::A
    coef::TCoefs
    it::IT
    ranges::RT
    itp::ITP
    sitp::SITP

    function InPlaceInterp(x::A, it::IT, ranges::RT) where {T, N, A<:AbstractArray{T,N}, IT, RT}
        Interpolations.check_ranges(it, axes(x), ranges)
        TC = Interpolations.tcoef(x)
        coef = padded_coefs(TC, x, it)
        TCoefs = typeof(coef)

        tw = Interpolations.tweight(x)
        itp = Interpolations.BSplineInterpolation(tw, coef, it, axes(x))
        sitp = Interpolations.scale(itp, ranges...)
        ITP = typeof(itp)
        SITP = typeof(sitp)
        return new{T, TC, N, A, TCoefs, IT, RT, ITP, SITP}(x, coef, it, ranges, itp, sitp)
    end
end

data(x::InPlaceInterp) = x.x
coefficients(x::InPlaceInterp) = x.coef
itpflag(x::InPlaceInterp) = x.it
ranges(x::InPlaceInterp) = x.ranges
size(x::InPlaceInterp) = size(data(x))


function update_interpolation!(x::InPlaceInterp)
    A = data(x)
    coefs = coefficients(x)
    it = itpflag(x)
    tw = Interpolations.tweight(A)

    indsA = axes(A)
    indspad = Interpolations.padded_axes(indsA, it)
    @assert indspad == axes(coefs) "coefs axes != padded_axes(x.x, x.it)"

    if indspad == indsA
        copyto!(coefs, A)
    else
        fill!(coefs, 0)
        Interpolations.ct!(coefs, indsA, A, indsA)
    end
    Interpolations.prefilter!(tw, coefs, it)
end

interpolation(x::InPlaceInterp) = x.itp
scaled_interpolation(x::InPlaceInterp) = x.sitp

# function interpolation(x::InPlaceInterp)
#     A = data(x)
#     tw = Interpolations.tweight(A)
#     return Interpolations.BSplineInterpolation(tw, coefficients(x), itpflag(x), axes(A))
# end
#
# function scaled_interpolation(x::InPlaceInterp)
#     itp = interpolation(x)
#     return Interpolations.scale(itp, ranges(x)...)
# end
