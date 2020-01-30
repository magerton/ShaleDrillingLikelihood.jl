function uview_col(x::SparseMatrixCSC, j::Integer)
    vals = nonzeros(x)
    rng = nzrange(x, j)
    return view(vals, rng)
end

"""
for each period `t`, we need to compute
    1. State transitions: Pr(sₜ₊₁ | sₜ)
    2. Expected value of Type I extreme value shocks
"""
# function update_sparse_state_transition!(simtmp::SimulationTmp,  uv::NTuple{2,<:Real}, z::NTuple{NZ,Real}, itypidx::NTuple{NI,Real}, state_if_no_drilling::Integer) where {NZ,NI}

function update_sparse_state_transition!(simtmp::SimulationTmp,
    obs::ObservationDrill, sim::SimulationDraw, theta)

    model = _model(obs)
    wp = statespace(model)
    P = Pprime(simtmp)

    vals = nonzeros(P)
    fill!(vals, 0)

    states_after_drilling = end_lrn(wp)+1 : end_inf(wp)
    states_can_start_from = flatten( (_x(obs), states_after_drilling, )  )

    for si in states_can_start_from

        obs_cf = ObservationDrill(model, ichars(obs), zchars(obs), 0, si)
        actions = actionspace(obs_cf)
        dp1sp = actions .+ 1

        local ubV = view(vdfull(simtmp), dp1sp)
        local actionprobs = uview_col(P, si)

        # if we can't drill
        if end_ex0(wp) < si <= end_lrn(wp) || si == end_inf(wp)
            @assert length(actions) == 1
            fill!(ubV, 0)
            fill!(actionprobs, 1)
        else
            for d in actions
                ubV[d+1] = full_payoff!(theta, d, obs_cf, theta, sim, false)
            end
        end

        # this is ok to do b/c logsumexp(x::Float64) = x
        lse = logsumexp!(actionprobs, ubV)
        isfinite(lse) || throw(error("lse = $lse with actionprobs = $actionprobs and ubV = $ubV"))
        eeps = lse - dot(actionprobs, ubV)
        isfinite(eeps) || throw(error("Eeps = $eeps with actionprobs = $actionprobs and ubV = $ubV and lse=$lse"))
        setindex!(Eeps(simtmp), eeps, si)
      end
end

struct LeaseCounterfactual{L<:DrillLease} <: AbstractObservationGroup
    data::L
end

zchars( x::LeaseCounterfactual) = zchars(DataDrill(x))

@delegate LeaseCounterfactual.data [jtstart, tstart, uniti, _regime, ichars]

firstindex(x::LeaseCounterfactual) = tstart(_data(x))   # tptr[j]
length(    x::LeaseCounterfactual) = length(zchars(x)) - jtstart(_data(x)) + 1
lastindex( x::LeaseCounterfactual) =  tstart(_data(x)) + length(x)-1
eachindex( x::LeaseCounterfactual) = firstindex(x) : lastindex(x)

first_state(x::LeaseCounterfactual) = _x(DataDrill(x), tstart(x))

function state_if_never_drilled(x::LeaseCounterfactual, t)
    model = _model(DataDrill(x))
    wp = statespace(model)
    return state_if_never_drilled(wp, first_state(x), t)
end

function getindex(x::LeaseCounterfactual, t)
    time_since_start = t - tstart(x)
    zt = jtstart(x) + time_since_start
    d = DataDrill(x)
    x0 = state_if_never_drilled(x,time_since_start)
    return ObservationDrill(_model(d), ichars(x), zchars(d,zt), 0, x0), zt
end
