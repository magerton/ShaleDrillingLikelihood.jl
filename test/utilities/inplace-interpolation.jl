module ShaleDrillingLikelihood_InterpolationTest

using ShaleDrillingLikelihood
using Interpolations
using InteractiveUtils
using Test
using BenchmarkTools

using ShaleDrillingLikelihood:
    update_interpolation!, interpolation, scaled_interpolation,
    InPlaceInterp

using Interpolations: DimSpec # , scale

using Base: OneTo

const SHOW_CODE_WARNTYPE = false

# something to interpolate
A_x1 = 1 : 0.1 : 10
A_x2 = 1 : 0.5 : 20
A_x3 = OneTo(2)

f(x1, x2, x3) = log(x1+x2) + x3
A = [f(x1,x2,x3) for x1 in A_x1, x2 in A_x2, x3 in A_x3]

# for scaling
ranges = (A_x1, A_x2, A_x3)
splinetype(r::AbstractRange) = BSpline(Quadratic(Free(OnGrid())))
splinetype(r::AbstractUnitRange) = NoInterp()

# itpflag
it = splinetype.(ranges)
it isa DimSpec{BSpline} where {N}

# generate a test case
test_itp = interpolate(A, it)
test_sitp = scale(test_itp, ranges...)

# our object
inplace_itp = InPlaceInterp(A, it, ranges)
myitp = interpolation(inplace_itp)
mysitp = scaled_interpolation(inplace_itp)

# don't check out before updating coefs (prefilter)
@test !(myitp == test_itp)
@test !(mysitp == test_sitp)

# but aftewards, they do
update_interpolation!(inplace_itp)
@test myitp == test_itp
@test test_sitp == mysitp

if SHOW_CODE_WARNTYPE
    @code_warntype interpolation(inplace_itp)
    @code_warntype scaled_interpolation(inplace_itp)
    @code_warntype Interpolations.gradient(mysitp, 1.25, 1.25, 1)
end

vec = A_x2[2:end-1] .+ 0.1
jnk = zero(vec)

@inline function f1(grad, itp, vec)
    broadcast!(v -> itp(1,v,2), grad, vec)
end

@inline function f2(grad, itp, vec)
    grad .= itp(1,vec,2)
end

if SHOW_CODE_WARNTYPE
    @code_warntype f1(jnk, mysitp, vec)
    @code_warntype f2(jnk, mysitp, vec)
end

# timing to figure out most efficient way to udpate
@btime f1($jnk, $mysitp, $vec)
@btime f2($jnk, $mysitp, $vec)

# @code_typed optimize=true  f(jnk, mysitp, vec)


end
