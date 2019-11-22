"Full-blown Dynamic discrete choice model"
struct DynamicDrillingModel{T<:Real, PF<:DrillReward, AUP<:AbstractUnitProblem, TT<:Tuple, AM<:AbstractMatrix{T}, AR<:StepRangeLen{T}}
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
        nz == prod(lengths.(zspace)) || throw(DimensionMismatch("zspace dim != ztransition dim"))

        return DynamicDrillingModel{T,APF,AUP,TT,AM,AR}(reward, discount, statespace, zspace, ztransition, psispace, anticipate_t1ev)
    end
end

reward(         x::DynamicDrillingModel) = x.reward
discount(       x::DynamicDrillingModel) = x.discount
statespace(     x::DynamicDrillingModel) = x.statespace
zspace(         x::DynamicDrillingModel) = x.zspace
ztransition(    x::DynamicDrillingModel) = x.ztransition
psispace(       x::DynamicDrillingModel) = x.psispace
anticipate_t1ev(x::DynamicDrillingModel) = x.anticipate_t1ev

# -----------------------------------------
# components of stuff
# -----------------------------------------

const DrillModel = Union{DynamicDrillingModel,DrillReward}

# access components
revenue(x::DrillReward) = x.revenue
drill(  x::DrillReward) = x.drill
extend( x::DrillReward) = x.extend
cost(   x::DrillReward) = drill(x)

@deprecate revenue(x::ObservationDrill) revenue(_model(x))
@deprecate drill(  x::ObservationDrill) drill(  _model(x))
@deprecate extend( x::ObservationDrill) extend( _model(x))
@deprecate extensioncost(x::DrillModel) extend(x)
@deprecate drillingcost( x::DrillModel) drill(x)

@deprecate flow(x::DynamicDrillingModel)          reward(x)
@deprecate β(x::DynamicDrillingModel)             discount(x)
@deprecate wp(x::DynamicDrillingModel)            statespace(x)
@deprecate _zspace(x::DynamicDrillingModel)       zspace(x)
@deprecate Πz(x::DynamicDrillingModel)            ztransition(x)
@deprecate _ψspace(x::DynamicDrillingModel)       psispace(x)
@deprecate anticipate_e(x::DynamicDrillingModel)  anticipate_t1ev(x)
