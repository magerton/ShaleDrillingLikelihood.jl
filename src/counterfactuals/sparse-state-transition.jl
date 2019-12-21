#
# """
# for each period `t`, we need to compute
#     1. State transitions: Pr(sₜ₊₁ | sₜ)
#     2. Expected value of Type I extreme value shocks
# """
# # function update_sparse_state_transition!(simtmp::SimulationTmp,  uv::NTuple{2,<:Real}, z::NTuple{NZ,Real}, itypidx::NTuple{NI,Real}, state_if_no_drilling::Integer) where {NZ,NI}
#
# function update_sparse_state_transition!(simtmp::SimulationTmp, obs::ObservationDrill, sim::SimulationDraw, theta)
#
#     model = _model(obs)
#     wp = statespace(model)
#     P = Pprime(simtimp)
#
#     vals = nonzeros(P)
#
#     states_after_drilling = end_ex0(wp)+1 : end_inf(wp)
#     states_can_start_from = flatten( (state_if_no_drilling, states_after_drilling, )  )
#
#     for si in states_can_start_from
#
#         obs_cf = ObservationDrill(model, ichars(obs), 0, si)
#
#         actions = actionspace(obs)
#         dp1sp = actions .+ 1
#
#         ubV = uview(vdfull(simtmp), dp1sp)
#         actionprobs = uview(vals, nzrange(P,si))
#
#         if end_ex0(wp) < si <= end_lrn(wp) || si == end_inf(wp)
#             fill!(ubV, 0)
#             fill!(actionprobs, 1)
#         else
#             @inbounds for d in actions
#                 ubV[d+1] = full_payoff(d, obs, theta, sim)
#             end
#         end
#
#         # this is ok to do b/c logsumexp(x::Float64) = x
#         lse = logsumexp!(actionprobs, ubV)
#         Eeps[si] = lse - dot(actionprobs, ubV)
#       end
# end

struct LeaseCounterfactual{L<:DrillLease} <: AbstractObservationGroup
    data::L
end

zchars( x::LeaseCounterfactual) = zchars(DataDrill(x))

@delegate LeaseCounterfactual.data [jtstart, tstart, uniti, _regime, ichars]

# jtstart(x::LeaseCounterfactual) = jtstart(_data(x))
# tstart( x::LeaseCounterfactual) = tstart( _data(x))
# uniti(  x::LeaseCounterfactual) = uniti(_data(x))
# _regime(x::LeaseCounterfactual) = _regime(_data(x))
# ichars( x::LeaseCounterfactual) = ichars(_data(x))

firstindex(x::LeaseCounterfactual) = tstart(_data(x))   # tptr[j]
_i(     x::LeaseCounterfactual) = _i(firstindex(x))
length(    x::LeaseCounterfactual) = length(zchars(x)) - jtstart(_data(x)) + 1
lastindex( x::LeaseCounterfactual) =  tstart(_data(x)) + length(x)-1
eachindex( x::LeaseCounterfactual) = firstindex(x) : lastindex(x)

first_state(x::LeaseCounterfactual) = _x(DataDrill(x), tstart(x))

function state_if_never_drilled(x::LeaseCounterfactual, t)
    model = _model(DataDrill(x))
    wp = statespace(model)
    state_if_never_drilled(wp, first_state(x), t)
end

function getindex(x::LeaseCounterfactual, t)
    time_since_start = t - tstart(x)
    zt = jtstart(x) + time_since_start
    d = DataDrill(x)
    x0 = state_if_never_drilled(x,t)
    return ObservationDrill(_model(d), ichars(x), zchars(d,zt), 0, x0)
end
