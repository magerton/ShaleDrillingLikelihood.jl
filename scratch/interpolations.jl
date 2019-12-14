using Interpolations
using Interpolations: tweight, tcoef, prefilter, copy_with_padding, prefilter!, BSplineInterpolation

A_x1 = 1 : 0.1 : 10
A_x2 = 1 : 0.5 : 20

f(x1, x2) = log(x1+x2)
A = [f(x1,x2) for x1 in A_x1, x2 in A_x2]

splinetype(r::AbstractRange) = BSpline(Quadratic(InPlace()))
splinetype(r::UnitRange) = BSpline(Constant())


b2 = BSpline(Constant())
b1 = BSpline(Quadratic(Free(OnGrid())))

it = (b1, b2,)
it isa NTuple{N,BSpline} where {N}


interpolate(A, it) # start here

# infer tw and tc, then
tw = tweight(A)
tc = tcoef(A)

# interpolate
@which interpolate(tw, tc, A, it)
# D:\software_libraries\julia\packages\Interpolations\0zZ6m\src\b-splines\b-splines.jl:132
# function interpolate(::Type{TWeights}, ::Type{TC}, A, it::IT) where {TWeights,TC,IT<:DimSpec{BSpline}}
#     Apad = prefilter(TWeights, TC, A, it)
#     BSplineInterpolation(TWeights, Apad, it, axes(A))
# end

# D:\software_libraries\julia\packages\Interpolations\0zZ6m\src\b-splines\prefiltering.jl:39
@which prefilter(tw, tc, A, it)
# function prefilter(::Type{TWeights}, ::Type{TC}, A::AbstractArray, it) where {TWeights,TC}
#     ret = copy_with_padding(TC, A, it)
#     prefilter!(TWeights, ret, it)
# end

# D:\software_libraries\julia\packages\Interpolations\0zZ6m\src\b-splines\prefiltering.jl:21
@which copy_with_padding(tc, A, it)
ret = copy_with_padding(tc, A, it)
size(ret)
# copy_with_padding(A, it) = copy_with_padding(eltype(A), A, it)
# function copy_with_padding(::Type{TC}, A, it::DimSpec{InterpolationType}) where {TC}
#     indsA = axes(A)
#     indspad = padded_axes(indsA, it)
#     coefs = padded_similar(TC, indspad)
#     if indspad == indsA
#         coefs = copyto!(coefs, A)
#     else
#         fill!(coefs, zero(TC))
#         ct!(coefs, indsA, A, indsA)
#     end
#     coefs
# end

@which BSplineInterpolation(tw, Apad, it, axes(A))
bsp = BSplineInterpolation(tw, Apad, it, axes(A))

Interpolations.padded_axes(axes(A), it)



function padded_coefs(::Type{TC}, A, it) where {TC}
    indsA = axes(A)
    indspad = Interpolations.padded_axes(indsA, it)
    return Interpolations.padded_similar(TC, indspad)
end
padded_coefs(A,it) = padded_coefs(tcoef(A), A, it)


sitp = scale(itp, A_x1, A_x2)

sitp(5., 10.) # exactly log(5 + 10)
sitp(5.6, 7.1) # approximately log(5.6 + 7.1)

struct EmaxCoefs{TC,IT<:DimSpec{BSpline},A1<:AbstractArray{T},A2<:AbstractArray{T}}
    EV::A1
    dEV::A2
    it::IT
    function EmaxCoefs(ev::A1, dev::A2, it::IT) where {TC,IT,A1<:AbstractArray{TC},A2<:AbstractArray{TC}}
        szdev = size(dev)
        (szdev[1:end-2]..., szdev[end]) == size(ev) || throw(DimensionMismatch())
        length(it)+1 == length(szdev) || throw(DimensionMismatch())
        new{T,IT,A1,A2}(ev, dev, it)
    end
end

EV(x::EmaxCoefs) = x.EV
dEV(x::EmaxCoefs) = x.dEV
ITEV(x::EmaxCoefs) = (x.it..., BSpline(Constant()))
ITdEV(x::EmaxCoefs) = (x.it..., BSpline(Constant()), BSpline(Constant()))

function EmaxCoefs(evs::DCDPEmax, it::Tuple)
    bspc = BSpline(Constant())
    ev_it = (it..., bspc, )
    dev_it = (it..., bspc, bspc,)

    ev_coef = padded_coefs(EV(evs), ev_it)
    dev_coef = padded_coefs(EV(evs), dev_it)

    return EmaxCoefs(ev_coef, dev_coef, it)
end

function copy_with_padding!(::Type{TC}, coefs, A, it::DimSpec{InterpolationType}) where {TC}
    indsA = axes(A)
    indspad = Interpolations.padded_axes(indsA, it)
    lengths.(indspad) == size(coefs)
    if indspad == indsA
        copyto!(coefs, A)
    else
        fill!(coefs, zero(TC))
        Interpolations.ct!(coefs, indsA, A, indsA)
    end
    return nothing
end


function update!(coefs::EmaxCoefs{TC}, evs::DCDPEmax, dograd) where {TC}
    TC == tcoef(EV(evs)) || throw(error())

    twev = tweights(EV(evs))
    copy_with_padding!(TC, EV(coefs), EV(evs), ITEV(coefs))
    prefilter!(twev, EV(coefs), it)

    if dograd
        twdev = tweights(dEV(evs))
        copy_with_padding!(TC, dEV(coefs), dEV(evs), ITdEV(coefs))
        prefilter!(twdev, dEV(coefs), dEV(evs), ITdEV(coefs))
    end
    return nothing
end



function interpolate(coefs::EmaxCoefs, evs::DCDPEmax)
    evtw = tweights(EV(evs))
    evax = axes(EV(evs))

    devtw = tweights(dEV(evs))
    devax = axes(dEV(evs))

    ev_itp = BSplineInterpolation(tw, EV(coefs), ITEV(coefs), ax)
    dev_itp = BSplineInterpolation(tw, dEV(coefs), ITdEV(coefs), ax)

    return ev_itp, dev_itp
end

function scale(coefs::EmaxCoefs, evs::DCDPEmax, ddm::DyanmicDrillintModel)
    ev, dev = interpolate(coefs, evs)
    ns = OneTo(length(statespace(ddm)))
    nk = OneTo(_nparm(reward(ddm)))
    sev = scale(ev, zspace(ddm)..., ns)
    sdev = scale(dev, zspace(ddm)..., nk, ns)
    return sev, sdev
end


# scaling 114
# indexint 32
