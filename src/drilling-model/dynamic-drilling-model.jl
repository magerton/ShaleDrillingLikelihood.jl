export DynamicDrillModel,
    reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev,
    DDM_NoVF

# -----------------------------------------
# Model
# -----------------------------------------

"Full-blown Dynamic discrete choice model"
struct DynamicDrillModel{T<:Real, PF<:DrillReward, AM<:AbstractMatrix{T},
    AUP<:AbstractUnitProblem, TT<:Tuple, AR<:StepRangeLen{T}, VF<:Union{AbstractValueFunction,Nothing}
} <: AbstractDynamicDrillModel
    reward::PF            # payoff function
    discount::T           # discount factor
    statespace::AUP       # structure of endogenous choice vars
    zspace::TT            # z-space (tuple)
    ztransition::AM       # transition for z
    psispace::AR          # ψspace = (u, ρu + sqrt(1-ρ²)*v)
    anticipate_t1ev::Bool # do we anticipate the ϵ shocks assoc w/ each choice?
    vf::VF                # value function
    tmpv::DCDPTmpVarsArray{T,AM}

    function DynamicDrillModel(
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

        tmpv = DCDPTmpVars(ntheta, nz, npsi, nd, ztransition)

        return new{T,APF,AM,AUP,TT,AR,VF}(reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev, vfout, tmpv)
    end
end

@deprecate DCDPEmax(args...) ValueFunctionArrayOnly(args...)

reward(         x::DynamicDrillModel) = x.reward
discount(       x::DynamicDrillModel) = x.discount
statespace(     x::DynamicDrillModel) = x.statespace
zspace(         x::DynamicDrillModel) = x.zspace
ztransition(    x::DynamicDrillModel) = x.ztransition
psispace(       x::DynamicDrillModel) = x.psispace
anticipate_t1ev(x::DynamicDrillModel) = x.anticipate_t1ev
value_function( x::DynamicDrillModel) = x.vf
DCDPTmpVars(    x::DynamicDrillModel) = x.tmpv

beta_1minusbeta(ddm::DynamicDrillModel) = discount(ddm) / (1-discount(ddm))

theta_drill_ρ(d::DynamicDrillModel, theta) = theta[_nparm(reward(d))]

function DynamicDrillModel(ddm::DynamicDrillModel, rwrd::DrillReward, wp = statespace(ddm))
    ddmnew = DynamicDrillModel(
        rwrd, discount(ddm), wp, zspace(ddm), ztransition(ddm),
        psispace(ddm), anticipate_t1ev(ddm)
    )
    return ddmnew
end

@deprecate DynamicDrillModel(
    rwrd::DrillReward, ddm::DynamicDrillModel, wp::AbstractStateSpace
    ) DynamicDrillModel(ddm, rwrd, wp)

# -----------------------------------------
# Outer constructors for VF from DDM
# -----------------------------------------

const DDM_NoVF       = DynamicDrillModel{T,APF,AM,AUP,TT,AR, Nothing}                  where {T,APF,AM,AUP,TT,AR}
const DDM_VFAO       = DynamicDrillModel{T,APF,AM,AUP,TT,AR, <:ValueFunctionArrayOnly} where {T,APF,AM,AUP,TT,AR}
const DDM_VF         = DynamicDrillModel{T,APF,AM,AUP,TT,AR, <:ValueFunction}          where {T,APF,AM,AUP,TT,AR}
const DDM_AbstractVF = DynamicDrillModel{T,APF,AM,AUP,TT,AR, <:AbstractValueFunction}  where {T,APF,AM,AUP,TT,AR}

DDM_NoVF(rwrd, beta, wp, z, ztrans, psi, t1ev) = DynamicDrillModel(rwrd, beta, wp, z, ztrans, psi, t1ev,  (args...) -> nothing)
DDM_NoVF(m::DynamicDrillModel) = DDM_NoVF(
        reward(m), discount(m), statespace(m), zspace(m), ztransition(m),
        psispace(m), anticipate_t1ev(m)  )

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


@inline function clamp_psi(m::DynamicDrillModel, psi)
    psis = psispace(m)
    return clamp(psi, first(psis), last(psis))
end

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

    psi_unsafe = _ψ(obs, sim)
    psi = clamp_psi(mod, psi_unsafe)
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



function initialize_x!(x, m::DynamicDrillModel, lease)
    s1 = 1
    s2 = end_ex1(statespace(m))+1
    x[1] = s1 # sample([s1,s2])
end

function update_x!(x, t, m::DynamicDrillModel, state, d)
    wp = statespace(m)
    @assert d in actionspace(wp,state)
    @assert state <= end_ex0(wp)+1 || state > end_lrn(wp)
    if t+1 <= length(x)
        x[t+1] = ssprime(wp, state, d)
    end
end


function ichars_sample(m::DynamicDrillModel, num_i)
    # geo, roy
    dist_geo = Normal(4.67, 0.33)
    dist_roy = [1/8, 1/6, 3/16, 1/5, 9/40, 1/4]
    geos = rand(dist_geo, num_i)
    roys = sample(dist_roy, num_i)
    return [(g,r) for (g,r) in zip(geos, roys)]
end
