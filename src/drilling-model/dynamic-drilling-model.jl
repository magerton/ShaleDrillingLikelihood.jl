export DynamicDrillingModel,
    reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev,
    ValueFunctionArrayOnly,
    ValueFunction,
    EV_scaled_itp,
   dEV_scaled_itp


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

const ObservationDynamicDrill = ObservationDrill{<:DynamicDrillingModel}

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
psispace(       x::DynamicDrillingModel) = x.psispace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev

beta_1minusbeta(ddm::DynamicDrillingModel) = discount(ddm) / (1-discount(ddm))


# -----------------------------------------
# Value Function Arrays
# -----------------------------------------

abstract type AbstractValueFunction end

"Value function arrays for dynamic model"
struct ValueFunctionArrayOnly{T<:Real} <: AbstractValueFunction
    EV::Array{T,3}
    dEV::Array{T,4}
    function ValueFunctionArrayOnly(EV, dEV)
         nz, npsi, nk, ns  = size(dEV)
        (nz, npsi,     ns) == size(EV)  || throw(DimensionMismatch("EV is $(size(EV)) but dEV is $(size(dEV))"))
        new{eltype(EV)}(EV, dEV)
    end
end

EV(x::ValueFunctionArrayOnly) = x.EV
dEV(x::ValueFunctionArrayOnly) = x.dEV

size(x::ValueFunctionArrayOnly) = size(dEV(x))

function fill!(x::ValueFunctionArrayOnly, y)
    fill!(EV(x), y)
    fill!(dEV(x), y)
end

function ValueFunctionArrayOnly(ddm::DynamicDrillingModel{T}) where {T}
    nz = size(ztransition(ddm), 1)
    nψ = length(psispace(ddm))
    nK = _nparm(reward(ddm))
    nS = length(statespace(ddm))

    dev = zeros(T, nz, nψ, nK, nS)
     ev = zeros(T, nz, nψ,     nS)
    return ValueFunctionArrayOnly(ev, dev)
end

@deprecate DCDPEmax(args...) ValueFunctionArrayOnly(args...)

"Value Function arrays with ability to interpolate"
struct ValueFunction{T<:Real, TC<:Real, N, NP1,
    EV_ITP <:InPlaceInterp{T, TC, N,   Array{T,N}  },
    DEV_ITP<:InPlaceInterp{T, TC, NP1, Array{T,NP1}}
} <:AbstractValueFunction

    EV::EV_ITP
    dEV::DEV_ITP

    function ValueFunction(ev::EV_ITP, dev::DEV_ITP) where {
        T<:Real, TC<:Real, N, NP1,
        EV_ITP <:InPlaceInterp{T, TC, N,   Array{T,N}  },
        DEV_ITP<:InPlaceInterp{T, TC, NP1, Array{T,NP1}}
    }
        size(ev)    == drop_second_to_last(size(dev)) || throw(DimensionMismatch())
        itpflag(ev) == drop_second_to_last(itpflag(dev)) || throw(error("itpflags are different"))
        ranges(ev)  == drop_second_to_last(ranges(dev)) || throw(error("ranges are different"))

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

 EV_scaled_itp(x::ValueFunction) = scaled_interpolation( EVobj(x))
dEV_scaled_itp(x::ValueFunction) = scaled_interpolation(dEVobj(x))


const DEFAULT_SPLINE = BSpline(Quadratic(Free(OnGrid())))
splinetype(r::AbstractRange, spline::Union{NoInterp,<:BSpline}=DEFAULT_SPLINE) = spline
splinetype(r::AbstractUnitRange, spline...) = NoInterp()

function ValueFunction(ddm::DynamicDrillingModel{T}, args...) where {T}
    nK = _nparm(reward(ddm))
    nS = length(statespace(ddm))

    dev_ranges = (zspace(ddm)..., psispace(ddm), OneTo(nK), OneTo(nS))
    dev_dims = length.(dev_ranges)

    ev_dims   = drop_second_to_last(dev_dims)
    ev_ranges = drop_second_to_last(dev_ranges)

    ev_it  = splinetype.( ev_ranges, args...)
    dev_it = splinetype.(dev_ranges, args...)

    ev = zeros(T, ev_dims)
    dev = zeros(T, dev_dims)

    EV  = InPlaceInterp( ev,  ev_it,  ev_ranges)
    dEV = InPlaceInterp(dev, dev_it, dev_ranges)

    return ValueFunction(EV, dEV)
end

# -----------------------------------------
# Value Function Arrays
# -----------------------------------------

"""
    discounted_dynamic_payoff!(grad, d, obs, sim, dograd)

compute `β * E[ V(z',ψ'x') | z,ψ,x]` and also gradient
"""
function discounted_dynamic_payoff!(grad, d::Integer, obs::ObservationDynamicDrill, sim, dograd)

    mod = _model(obs)
    rwrd = reward(mod)
    beta = discount(mod)
    nk = length(grad)
    nk == _nparm(rwrd) || throw(DimensionMismatch())

    vf = value_function(mod)
    vf_sitp = EV_scaled_itp(vf)

    psi = _ψ(obs, sim)
    sp = sprime(statespace(mod), _x(obs), d)
    z = zchars(obs)

    if dograd
        dvf_sitp = dEV_scaled_itp(vf)
        @inbounds @fastmath @simd for k in OneTo(nk)
            grad[k] += beta * dvf_sitp(z..., psi, k, sp)
        end
        dpsi = last(Interpolations.gradient(vf_sitp, z..., psi, k, sp))
        grad[idx_ρ(rwrd)] += dpsi * beta * _dψdθρ(obs, sim)
    end

    return beta * vf_sitp(z..., psi, sp)
end


function full_payoff!(grad, d::Integer, obs::ObservationDynamicDrill, theta, sim, dograd)
    rwrd = reward(_model(obs))
    static_payoff  = flow!(grad, rwrd, d, obs, theta, sim, dograd)
    dynamic_payoff = discounted_dynamic_payoff!(grad, obs, sim, dograd)
    return static_payoff + dynamic_payoff
end
