@GenGlobal g_SharedSimulations
@GenGlobal g_SimulationPrimitives
@GenGlobal g_BaseDataSetofSets
@GenGlobal g_SimulationDrawsMatrix

# ------------------------------------------------------
# sparse state transition matrix
# ------------------------------------------------------

function sparse_state_transition(wp::AbstractUnitProblem)
     nS = length(wp)
     rowval = [ssprime(wp, si, d) for si in 1:nS for d in actionspace(wp, si)]
     colptr = cumsum(vcat(1, [_dmax(wp,si)+1 for si in 1:nS]))
     nzval = zeros(Float64, length(rowval))

     return SparseMatrixCSC(nS, nS, colptr, rowval, nzval)
end

sparse_state_transition(ddm::DynamicDrillModel) = sparse_state_transition(statespace(ddm))

# ------------------------------------------------------
# Temporary variables for simulations
# ------------------------------------------------------

"""
Holds temp variables needed for each simulation of a section `i` given
    draw number `m` from shared Halton arrays `uvs`
"""
struct SimulationTmp <: AbstractTmpVars
     Pprime::SparseMatrixCSC{Float64,Int}
     @addStructFields Vector{Float64} sa sb qm LLJ profit surplus drillcost revenue vdfull Eeps

     function SimulationTmp(Pprime, sa, sb, qm, LLJ, profit, surplus, drillcost, revenue, vdfull, Eeps)
         length(sa) == length(sb) == length(Eeps) == checksquare(Pprime) ||
            throw(DimensionMismatch())

         length(profit) == length(surplus) == length(drillcost) == length(revenue) ==
            length(vdfull) || throw(DimensionMismatch())

        return new(Pprime, sa, sb, qm, LLJ, profit, surplus, drillcost, revenue, vdfull, Eeps)
     end
end

@getFieldFunction SimulationTmp Pprime sa sb qm LLJ profit surplus drillcost revenue vdfull Eeps

function SimulationTmp(data::DataDrill, M::Integer)
    m = _model(data)

    Pprime = sparse_state_transition(statespace(m))

    nS = length(statespace(m))
    sa = Vector{Float64}(undef, nS)
    sb = similar(sa)
    Eeps = similar(sa)
    qm = Vector{Float64}(undef, M)

    LLJ = Vector{Float64}(undef, maxj1length(data))

    profit     = zeros(Float64, _dmax(statespace(m))+1)
    surplus    = similar(profit)
    drillcost  = similar(profit)
    revenue    = similar(profit)
    vdfull     = similar(profit)

     return SimulationTmp(Pprime, sa, sb, qm, LLJ, profit, surplus, drillcost, revenue, vdfull, Eeps)
end

"""
    isodd(t) ? (sa,sb) : (sb,sa)

`tday` holds our current state in `t`
`tmrw` holds the distribution of our state in `t+1` given prices
we switch between `sa` and `sb` to avoid allocating...
"""

function today_tomorrow(x::SimulationTmp, t::Integer)
    if isodd(t)
        return sa(x), sb(x)
    else
        return sb(x), sa(x)
    end
end

SimulationTmp(data::DataSetofSets, M) = SimulationTmp(drill(data), _num_sim(M))

num_states(x::SimulationTmp) = length(sa(x))

function reset!(x::SimulationTmp, s0::Integer)
    fill!(sa(x), 0)
    fill!(sb(x), 0)
    setindex!(sa(x), 1, s0)
    setindex!(sb(x), 1, s0)
    return nothing
end

function update!(simtmp::SimulationTmp, obs::ObservationDrill, sim::SimulationDraw, θ)
    update_sparse_state_transition!(simtmp, obs, sim, θ)

    m = _model(obs)
    wp = statespace(m)
    nS = length(wp)
    dograd = false
    newobs = ObservationDrill(m, ichars(obs), zchars(obs), 0, nS)

    x = reward(m)
    surp = NoRoyaltyProblem(x)

    # components of payoff. NOTE that these DO NOT depend on the deterministic state... just zₜ
    for d in actionspace(wp)

        c   = flow!(vw_cost(   x, θ), cost(   x), d, obs, vw_cost(   x, θ), sim, dograd)
        r   = flow!(vw_revenue(x, θ), revenue(x), d, obs, vw_revenue(x, θ), sim, dograd)
        pft = flow!(θ,                        x,  d, obs, θ,                sim, dograd)
        s   = flow!(θ,                     surp,  d, obs, θ,                sim, dograd)

        # assumes extension & eur are fixed
        setindex!( revenue(  simtmp), c  , d+1)  # (1-r)rev - 0
        setindex!( drillcost(simtmp), r  , d+1)  #    0     - drillcost(d)
        setindex!( profit(   simtmp), pft, d+1)  # (1-r)rev - drillcost(d)
        setindex!( surplus(  simtmp), s  , d+1)  #      rev - drillcost(d)
    end

    Q = eur_kernel(revenue(x), 0, obs, vw_revenue(x,θ), sim)
    ext = extensioncost(extend(x), vw_extend(x,θ))

    return Q, ext
end

# ----------------------
# Holds simulations
# ----------------------

"What kind of simulation to run given datastruct"
struct SimulationPrimitives{D<:DataSetofSets,R}
    data::D
    sim::SimulationDraws{Float64,2,Matrix{Float64}}
    Tstop::Int
    theta::Vector{Float64}
    simtmp::SimulationTmp
    sharedsim::SharedSimulations{R}

    function SimulationPrimitives(
        data::D, sim, Tstop, theta, sharedsim::SharedSimulations{R}
    ) where {D<:DataSetofSets, R}
        _nparm(data) == length(theta) || throw(DimensionMismatch("data & theta"))
        num_i(data) == num_i(sim) || throw(DimensionMismatch("num_i"))
        0 < Tstop || throw(DomainError(Tstop))
        drilldat = drill(data)
        Tstop <= length(zchars(drilldat)) || throw(DomainError(Tstop))
        new{D,R}(data, sim, Tstop, theta, SimulationTmp(data, sim), sharedsim)
    end
end

@noinline function SimulationPrimitives(
    data::DataSetofSets, sim::SimulationDrawsMatrix,
    rwrd::AbstractStaticPayoff, wp::AbstractStateSpace, Tstop, theta, sharedsim
    )

    datanew = DataSetofSets(data, DataDrill(drill(data), rwrd, wp))
    return SimulationPrimitives(datanew, sim, Tstop, theta, sharedsim)
end

function SimulationPrimitives(rwrd, wp, Tstop, theta)
    data = get_g_BaseDataSetofSets()
    sim = get_g_SimulationDrawsMatrix()
    sharedsim = get_g_SharedSimulations()
    return SimulationPrimitives(data, sim, rwrd, wp, Tstop, theta, sharedsim)
end

function set_g_SimulationPrimitives(rwrd, wp, Tstop, theta)
    simprim = SimulationPrimitives(rwrd, wp, Tstop, theta)
    set_g_SimulationPrimitives(simprim)
end

SimulationTmp(x::SimulationPrimitives) = x.simtmp
SharedSimulations(x::SimulationPrimitives) = x.sharedsim
SimulationDraws(x::SimulationPrimitives) = x.sim
SimulationDrawsMatrix(x::SimulationPrimitives) = SimulationDraws(x)
SimulationDrawsVector(x::SimulationPrimitives, i) =
    view(SimulationDrawsMatrix(x), i)

@getFieldFunction SimulationPrimitives Tstop theta data

split_thetas(x::SimulationPrimitives) = split_thetas(data(x), theta(x))
theta_drill(x::SimulationPrimitives) = first(split_thetas(x))
