
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

SharedSimulations(data::DataDrill) = SharedSimulations(workers(), data)

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
