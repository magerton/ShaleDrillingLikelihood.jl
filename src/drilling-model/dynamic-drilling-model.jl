export DynamicDrillingModel,
    reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev,
    ValueFunctionArrayOnly,
    ValueFunction,
    EV_scaled_itp,
   dEV_scaled_itp


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
      check_a_eq_b_sans_2nd_to_last(size(EV), size(dEV))  || throw(DimensionMismatch("EV is $(size(EV)) but dEV is $(size(dEV))"))
      new{eltype(EV)}(EV, dEV)
   end
end

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
# Model
# -----------------------------------------

"Full-blown Dynamic discrete choice model"
struct DynamicDrillingModel{T<:Real, PF<:DrillReward, AM<:AbstractMatrix{T}, AUP<:AbstractUnitProblem, TT<:Tuple, AR<:StepRangeLen{T}, VF<:Union{AbstractValueFunction,Nothing}} <: AbstractDrillModel
    reward::PF            # payoff function
    discount::T           # discount factor
    statespace::AUP       # structure of endogenous choice vars
    zspace::TT            # z-space (tuple)
    ztransition::AM       # transition for z
    psispace::AR          # ψspace = (u, ρu + sqrt(1-ρ²)*v)
    anticipate_t1ev::Bool # do we anticipate the ϵ shocks assoc w/ each choice?
    vf::VF                # value function

    function DynamicDrillingModel(
      reward::APF, discount::T, statespace::AUP, zspace::TT, ztransition::AM,
      psispace::AR, anticipate_t1ev, vf::Union{Type,Function}=ValueFunction
    ) where {
        T,N, APF, AUP, TT<:NTuple{N,AbstractRange},
        AM, AR
    }
        nz = checksquare(ztransition)
        npsi = length(psispace)
        nS = length(statespace)
        nd = length(actionspace(statespace))
        ntheta = _nparm(reward)

        0 < discount < 1 || throw(DomainError(discount))
        nz == prod(length.(zspace)) || throw(DimensionMismatch("zspace dim != ztransition dim"))

        vfout = vf(reward, discount, statespace, zspace, ztransition, psispace)
        VF = typeof(vfout)

        return new{T,APF,AM,AUP,TT,AR,VF}(reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev, vfout)
    end
end

const ObservationDynamicDrill = ObservationDrill{<:DynamicDrillingModel}

@deprecate DCDPEmax(args...) ValueFunctionArrayOnly(args...)

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
psispace(       x::DynamicDrillingModel) = x.psispace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev
value_function( x::DynamicDrillingModel) = x.vf

beta_1minusbeta(ddm::DynamicDrillingModel) = discount(ddm) / (1-discount(ddm))

theta_drill_ρ(d::DynamicDrillingModel, theta) = theta[_nparm(reward(d))]

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

# -----------------------------------------
# Outer constructors for VF from DDM
# -----------------------------------------

const DDM_NoVF = DynamicDrillingModel{T,APF,AM,AUP,TT,AR,Nothing} where {T,APF,AM,AUP,TT,AR}
const DDM_VFAO = DynamicDrillingModel{T,APF,AM,AUP,TT,AR,<:ValueFunctionArrayOnly} where {T,APF,AM,AUP,TT,AR}
const DDM_VF   = DynamicDrillingModel{T,APF,AM,AUP,TT,AR,<:ValueFunction} where {T,APF,AM,AUP,TT,AR}
const DDM_AbstractVF = DynamicDrillingModel{T,APF,AM,AUP,TT,AR,<:AbstractValueFunction} where {T,APF,AM,AUP,TT,AR}

ValueFunctionArrayOnly(ddm::DDM_NoVF) = ValueFunctionArrayOnly(reward(ddm), discount(ddm), statespace(ddm), zspace(ddm), ztransition(ddm), psispace(ddm))
ValueFunction(         ddm::DDM_NoVF) = ValueFunction(         reward(ddm), discount(ddm), statespace(ddm), zspace(ddm), ztransition(ddm), psispace(ddm))

ValueFunctionArrayOnly(ddm::DDM_VFAO) = value_function(ddm)
ValueFunctionArrayOnly(ddm::DDM_VF) = ValueFunctionArrayOnly(value_function(ddm))

ValueFunction(ddm::DDM_VF) = value_function(ddm)
function ValueFunction(ddm::DDM_VFAO)
    vfao = value_function(ddm)
    ev = EV(vfao)
    dev = dEV(vfao)
    return ValueFunction(ev, dev, reward(ddm), discount(ddm), statespace(ddm), zspace(ddm), ztransition(ddm), psispace(ddm))
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
        @inbounds @simd for k in OneTo(nk)
            grad[k] += beta * dvf_sitp(z..., psi, k, sp)
        end
        dpsi = last(Interpolations.gradient(vf_sitp, z..., psi, sp)) # FIXME
        grad[idx_ρ(rwrd)] += dpsi * beta * _dψdθρ(obs, sim)
    end

    return beta * vf_sitp(z..., psi, sp)
end


function full_payoff!(grad, d::Integer, obs::ObservationDynamicDrill, theta, sim, dograd)
    rwrd = reward(_model(obs))
    fill!(grad, 0) # FIXME!! 
    static_payoff  = flow!(grad, rwrd, d, obs, theta, sim, dograd)
    if dograd
        grad[idx_ρ(rwrd)] += flowdψ(rwrd, d, obs, theta, sim) * _dψdθρ(obs, sim)
    end
    dynamic_payoff = discounted_dynamic_payoff!(grad, d, obs, sim, dograd)
    return static_payoff + dynamic_payoff
end


# -----------------------------------------
# for data generation
# -----------------------------------------



function initialize_x!(x, m::DynamicDrillingModel, lease)
    x[1] = 1
end

function update_x!(x, t, m::DynamicDrillingModel, state, d)
    if t+1 <= length(x)
        x[t+1] = ssprime(statespace(m), state, d)
    end
end



function ichars_sample(m::DynamicDrillingModel, num_i)
    # geo, roy
    dist_geo = Normal(4.67, 0.33)
    dist_roy = [1/8, 1/6, 3/16, 1/5, 9/40, 1/4]
    geos = rand(dist_geo, num_i)
    roys = sample(dist_roy, num_i)
    return [(g,r) for (g,r) in zip(geos, roys)]
end
