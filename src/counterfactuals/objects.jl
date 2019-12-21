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
    N = num_i(data)

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

SimulationTmp(data::DataSetofSets, M) = SimulationTmp(drill(data), _num_sim(M))

# ----------------------
# Holds simulations
# ----------------------

"What kind of simulation to run given datastruct"
struct SimulationPrimitives{D<:DataSetofSets}
    data::D
    sim::SimulationDraws{Float64,2,Matrix{Float64}}
    Tstop::Int
    theta::Vector{Float64}
    simtmp::SimulationTmp

    function SimulationPrimitives(
        data::D, sim, Tstop, theta, simtmp
    ) where {D<:DataSetofSets}
        _nparm(data) == length(theta) || throw(DimensionMismatch("data & theta"))
        num_i(data) == num_i(sim) || throw(DimensionMismatch("num_i"))
        0 < Tstop || throw(DomainError(Tstop))
        drilldat = drill(data)
        Tstop <= length(zchars(drilldat)) || throw(DomainError(Tstop))
        new{D}(data, sim, Tstop, theta, SimulationTmp(data, sim))
    end
end

@noinline function SimulationPrimitives(
    data::DataSetofSets, sim::SimulationDrawsMatrix,
    rwrd::AbstractStaticPayoff, wp::AbstractStateSpace, Tstop, theta
    )

    datanew = DataSetofSets(data, DataDrill(drill(data), rwrd, wp))
    return SimulationPrimitives(datanew, sim, Tstop, theta)
end

function SimulationPrimitives(rwrd, wp, Tstop, theta)
    data = get_g_BaseDataSetofSets()
    sim = get_g_SimulationDrawsMatrix()
    return SimulationPrimitives(data, sim, rwrd, wp, Tstop, theta)
end

function set_g_SimulationPrimitives(rwrd, wp, Tstop, theta)
    simprim = SimulationPrimitives(rwrd, wp, Tstop, theta)
    set_g_SimulationPrimitives(simprim)
end

# ----------------------
# Holds simulations
# ----------------------

type_to_sym(x) = x |> typeof |> Symbol |> string

const SimulationList = Tuple{DrillReward,AbstractUnitProblem,Vector}

"Create `DataFrame` of info given vector of `simulationPrimitives`"
function simulationPrimitives_information(simulations::Vector{<:SimulationList})
    df = DataFrame()
    df[!, :i] = 1:length(simulations)

    df[!, :tech]        = CategoricalArray( map(s -> s |> getindex(1) |> revenue |> tech    |> type_to_sym, simulations) )
    df[!, :tax ]        = CategoricalArray( map(s -> s |> getindex(1) |> revenue |> tax     |> type_to_sym, simulations) )
    df[!, :tech]        = CategoricalArray( map(s -> s |> getindex(1) |> revenue |> learn   |> type_to_sym, simulations) )
    df[!, :tech]        = CategoricalArray( map(s -> s |> getindex(1) |> revenue |> royalty |> type_to_sym, simulations) )

    df[!, :state_space] = CategoricalArray( map(s -> s |> getindex(2)                       |> type_to_sym, simulations) )

    df[!, :theta]       = CategoricalArray( map(s -> s |> getindex(3)  |> round(; digits=3) |> string     , simulations) )
    return df
end

"construct `DataFrame`s to hold simulations"
function dataFrames_from_simulationPrimitives(simulations::Vector{<:SimulationList}, data::DataDrill, Tstop)

    nSim = length(simulations)
    nT = length(zchars(data))
    timestamps = _timestamp(zchars(data))
    nI = num_i(data)

    Dmx = _Dmax(getindex(first(simulations), 2))

    df_d_iter = product(1:nT,        1:nI, 1:nSim)
    df_D_iter = product(Tstop:Tstop, 1:nI, 1:nSim)

    ndf_d = length(df_d_iter)
    ndf_D = length(df_D_iter)

    df_d = DataFrame(
        sim     = vec(collect(Int32(ns)    for (t, i, ns,) in df_d_iter)),
        i       = vec(collect(Int32(i)     for (t, i, ns,) in df_d_iter)),
        mondate = vec(collect(timestamps[t] for (t, i, ns,) in df_d_iter)),
    )
    vars = [
        :d0, :d1, :d0psi, :d1psi, :d0eur, :d1eur, :d0eursq, :d1eursq, :d0eurcub, :d1eurcub,
        :epsdeq1, :epsdgt1, :Prdeq1, :Prdgt1,
        :Eeps, :profit, :surplus, :revenue, :drillcost, :extension,
    ]
    for v in vars
        df_d[!, v] = zeros(Float64, ndf_d)
    end

    df_D = DataFrame(
        sim     = vec(collect(Int32(ns)     for (t, i, ns,) in df_D_iter)),
        i       = vec(collect(Int32(i)      for (t, i, ns,) in df_D_iter)),
        mondate = vec(collect(timestamps[t]  for (t, i, ns,) in df_D_iter)),
    )
    for D in 0:Dmx
        df_D[!, Symbol("D$(D)")] = zeros(Float64, ndf_D)
    end

    return (df_d, df_D,)
end

# ----------------------
# Holds simulations
# ----------------------

"Container holds simulations of all sections"
struct SharedSimulations{R<:AbstractRange}
    @addStructFields(SharedMatrix{Float64}, d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub)
    @addStructFields(SharedMatrix{Float64}, epsdeq1, epsdgt1, Prdeq1, Prdgt1)
    @addStructFields(SharedMatrix{Float64}, Eeps, profit, surplus, revenue, drillcost, extension)
    @addStructFields(SharedMatrix{Float64}, D_at_T)
    zchars_time::R
end


"Create `SharedSimulations`"
function SharedSimulations(pids::Vector{<:Integer}, nT::Integer, N::Integer, Dmax::Integer, zchars_time::AbstractRange)

    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub)
    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), epsdeq1, epsdgt1, Prdeq1, Prdgt1)
    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), Eeps, profit, surplus, revenue, drillcost, extension)

    D_at_T = SharedMatrix{Float64}(Dmax, N, pids=pids)

    return SharedSimulations(
        d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub,
        epsdeq1, epsdgt1, Prdeq1, Prdgt1,
        Eeps, profit, surplus, revenue, drillcost, extension,
        D_at_T,
        zchars_time
    )
end

function SharedSimulations(pids, data::DataDrill)
    nT = length(zchars(data))
    N = num_i(data)
    Dmax = _Dmax(statespace(_model(data)))
    zchars_time = _timestamp(zchars(data))
    return SharedSimulations(pids, nT, N, Dmax, zchars_time)
end

SharedSimulations(data) = SharedSimulations(workers(), data)

# ----------------------
# Holds simulations
# ----------------------

"update `df_d` and `df_D` with simulation number `sim`, held in `drillsim`"
function update_sim_dataframes_from_simdata!(df_d::DataFrame, df_D::DataFrame, sim::Integer, drillsim::SharedSimulations)

    is_this_simulation_d = df_d[!,:sim] .== sim
    is_this_simulation_D = df_D[!,:sim] .== sim

    sum(is_this_simulation_d) == 0 && throw(error("no observations selected to update"))

    # update period-t simulations
    vars = [
        :d0, :d1, :d0psi, :d1psi, :d0eur, :d1eur, :d0eursq, :d1eursq, :d0eurcub, :d1eurcub,
        :epsdeq1, :epsdgt1, :Prdeq1, :Prdgt1,
        :Eeps, :profit, :surplus, :revenue, :drillcost, :extension,
    ]
    for v in vars
        simfld = getfield(drillsim, v)::SharedMatrix{Float64}
        df_d[is_this_simulation_d, v] = vec(sdata(simfld))
    end

    # update final period simulations
    df_D[is_this_simulation_D, :D0] = 1 .- vec(sum(sdata(drillsim.D_at_T);dims=1))
    for D in 1:size(drillsim.D_at_T, 1)
        @views df_D[is_this_simulation_D, Symbol("D$(D)")] = sdata(drillsim.D_at_T)[D,:]
    end
end
