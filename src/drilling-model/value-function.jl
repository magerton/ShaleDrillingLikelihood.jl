export ValueFunctionArrayOnly,
    ValueFunction,
    EV_scaled_itp,
   dEV_scaled_itp

"check if dims(a) = dims(b)[1:end-2, end]"
function check_a_eq_b_sans_2nd_to_last(a::Tuple, b::Tuple)
   length(b) == length(a)+1 || return false
   b[1:end-2] == a[1:end-1] || return false
   b[end] == a[end]         || return false
   return true
end


abstract type AbstractValueFunction end

# -----------------------------------------
# Value Function Arrays
# -----------------------------------------

"Value function arrays"
struct ValueFunctionArrayOnly{T<:Real} <: AbstractValueFunction
   EV::Array{T,3}
   dEV::Array{T,4}
   function ValueFunctionArrayOnly(EV, dEV)
      check_a_eq_b_sans_2nd_to_last(size(EV), size(dEV))  ||
        throw(DimensionMismatch("EV is $(size(EV)) but dEV is $(size(dEV))"))
      new{eltype(EV)}(EV, dEV)
   end
end

"Array suitable for VF iterations"
EV(  x::ValueFunctionArrayOnly) = x.EV
dEV( x::ValueFunctionArrayOnly) = x.dEV
size(x::ValueFunctionArrayOnly) = size(dEV(x))

function fill!(x::ValueFunctionArrayOnly, y)
    fill!(EV(x), y)
    fill!(dEV(x), y)
end

# -----------------------------------------
# Value Function Array w/ interpolation info
# -----------------------------------------

"Value Function arrays with ability to interpolate"
struct ValueFunction{T<:Real, TC<:Real, N, NP1,
    EV_ITP <:InPlaceInterp{T, TC, N,   Array{T,N}  },
    DEV_ITP<:InPlaceInterp{T, TC, NP1, Array{T,NP1}}
} <: AbstractValueFunction

    EV::EV_ITP
    dEV::DEV_ITP

    function ValueFunction(ev::EV_ITP, dev::DEV_ITP) where {
        T<:Real, TC<:Real, N, NP1,
        EV_ITP <:InPlaceInterp{T, TC, N,   Array{T,N}  },
        DEV_ITP<:InPlaceInterp{T, TC, NP1, Array{T,NP1}}
    }

        check_a_eq_b_sans_2nd_to_last(size(ev), size(dev))       || throw(DimensionMismatch())
        check_a_eq_b_sans_2nd_to_last(itpflag(ev), itpflag(dev)) || throw(error("itpflags are different"))
        check_a_eq_b_sans_2nd_to_last(ranges(ev), ranges(dev))   || throw(error("ranges are different"))

        return new{T,TC,N,NP1,EV_ITP,DEV_ITP}(ev, dev)
    end
end

EVobj( x::ValueFunction) = x.EV
dEVobj(x::ValueFunction) = x.dEV

function EV(x::ValueFunction)
    v = data(EVobj(x))
    nd = ndims(v)
    return reshape(v, :, size(v, nd-1), size(v, nd))
end

function dEV(x::ValueFunction)
    v = data(dEVobj(x))
    nd = ndims(v)
    return reshape(v, :, size(v, nd-2), size(v, nd-1), size(v, nd))
end

ValueFunctionArrayOnly(x::ValueFunction) = ValueFunctionArrayOnly(EV(x), dEV(x))

function fill!(x::ValueFunction, y)
    fill!(data( EVobj(x)), y)
    fill!(data(dEVobj(x)), y)
end

function update_interpolation!(x::ValueFunction, dograd)
    dograd && update_interpolation!(dEVobj(x))
    update_interpolation!(EVobj(x))
end

update_interpolation!(x...) = nothing
value_function(x...) = nothing


 EV_scaled_itp(x::ValueFunction) = scaled_interpolation( EVobj(x))
dEV_scaled_itp(x::ValueFunction) = scaled_interpolation(dEVobj(x))


# -----------------------------------------
# Outer constructors for VF
# -----------------------------------------

NoValueFunction(args...) = nothing

function ValueFunctionArrayOnly(reward, discount::T, statespace, zspace, ztransition, psispace, args...) where {T<:Real}
    nz = size(ztransition, 1)
    nψ = length(psispace)
    nK = _nparm(reward)
    nS = length(statespace)

    dev = zeros(T, nz, nψ, nK, nS)
     ev = zeros(T, nz, nψ,     nS)
    return ValueFunctionArrayOnly(ev, dev)
end

const DEFAULT_SPLINE = BSpline(Quadratic(Free(OnGrid())))
splinetype(r::AbstractRange, spline::Union{NoInterp,<:BSpline}=DEFAULT_SPLINE) = spline
splinetype(r::UnitRange, spline... ) = BSpline(Constant())
splinetype(r::AbstractUnitRange, spline...) = NoInterp()

function ValueFunction(ev, dev, reward, discount, statespace, zspace, ztransition, psispace, args...)
    nK = _nparm(reward)
    nS = length(statespace)
    nψ = length(psispace)
    nz = length.(zspace)

    ev_ranges  = (zspace..., psispace,            OneTo(nS))
    dev_ranges = (zspace..., psispace, OneTo(nK), OneTo(nS))

    ev_it  = splinetype.( ev_ranges, args...)
    dev_it = splinetype.(dev_ranges, args...)

    ev  = reshape(ev,  nz..., nψ,    nS)
    dev = reshape(dev, nz..., nψ, nK, nS)

    EV  = InPlaceInterp( ev,  ev_it,  ev_ranges)
    dEV = InPlaceInterp(dev, dev_it, dev_ranges)

    return ValueFunction(EV, dEV)
end

function ValueFunction(reward, discount::T, statespace, zspace, ztransition, psispace, args...) where {T<:Real}
    nK = _nparm(reward)
    nS = length(statespace)
    nψ = length(psispace)
    nz = length.(zspace)
    ev  = zeros(T, nz..., nψ,    nS)
    dev = zeros(T, nz..., nψ, nK, nS)
    return ValueFunction(ev, dev, reward, discount, statespace, zspace, ztransition, psispace, args...)
end
