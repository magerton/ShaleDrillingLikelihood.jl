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
function dataFramesForSimulations(simulations::Vector{<:SimulationList}, data::DataDrill, Tstop)

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

@deprecate dataFrames_from_simulationPrimitives(args...) dataFramesForSimulations(args...)

# ----------------------
# Holds simulations
# ----------------------

"update `df_d` and `df_D` with simulation number `sim`, held in `drillsim`"
function update_sim_dataframes_from_simdata!(df_d::DataFrame, df_D::DataFrame, drillsim::SharedSimulations, sim::Integer)

    is_this_simulation_d = df_d[!,:sim] .== sim
    is_this_simulation_D = df_D[!,:sim] .== sim

    sum(is_this_simulation_d) == 0 && throw(error("no observations selected to update"))

    # update period-t simulations
    df_d[is_this_simulation_d, :d0       ] = vec(sdata(drillsim.d0))
    df_d[is_this_simulation_d, :d1       ] = vec(sdata(drillsim.d1))
    df_d[is_this_simulation_d, :d0psi    ] = vec(sdata(drillsim.d0psi))
    df_d[is_this_simulation_d, :d1psi    ] = vec(sdata(drillsim.d1psi))
    df_d[is_this_simulation_d, :d0eur    ] = vec(sdata(drillsim.d0eur))
    df_d[is_this_simulation_d, :d1eur    ] = vec(sdata(drillsim.d1eur))
    df_d[is_this_simulation_d, :d0eursq  ] = vec(sdata(drillsim.d0eursq))
    df_d[is_this_simulation_d, :d1eursq  ] = vec(sdata(drillsim.d1eursq))
    df_d[is_this_simulation_d, :d0eurcub ] = vec(sdata(drillsim.d0eurcub))
    df_d[is_this_simulation_d, :d1eurcub ] = vec(sdata(drillsim.d1eurcub))
    df_d[is_this_simulation_d, :epsdeq1  ] = vec(sdata(drillsim.epsdeq1))
    df_d[is_this_simulation_d, :epsdgt1  ] = vec(sdata(drillsim.epsdgt1))
    df_d[is_this_simulation_d, :Prdeq1   ] = vec(sdata(drillsim.Prdeq1))
    df_d[is_this_simulation_d, :Prdgt1   ] = vec(sdata(drillsim.Prdgt1))
    df_d[is_this_simulation_d, :Eeps     ] = vec(sdata(drillsim.Eeps))
    df_d[is_this_simulation_d, :profit   ] = vec(sdata(drillsim.profit))
    df_d[is_this_simulation_d, :surplus  ] = vec(sdata(drillsim.surplus))
    df_d[is_this_simulation_d, :revenue  ] = vec(sdata(drillsim.revenue))
    df_d[is_this_simulation_d, :drillcost] = vec(sdata(drillsim.drillcost))
    df_d[is_this_simulation_d, :extension] = vec(sdata(drillsim.extension))

    # update final period simulations
    df_D[is_this_simulation_D, :D0] = 1 .- vec(sum(sdata(drillsim.D_at_T);dims=1))
    for D in 1:size(drillsim.D_at_T, 1)
        @views df_D[is_this_simulation_D, Symbol("D$(D)")] = sdata(drillsim.D_at_T)[D,:]
    end
end



function doSimulations(datafull::DataSetofSets, simlist::Vector{<:SimulationList}, Tstop, M)

    N = num_i(data_bare)
    wrkrs = CachingPool(workers())

    # generate smaller DataDrill to transfer to workers
    datadrill = drill(datafull)
    ddm_novf = DDM_NoVF(_model(datadrill))
    datadrill_dso = DataDrillStartOnly(datadrill)
    datadrill_bare= DataDrill(data_dso, ddm_novf)

    # DataFull set for transfer
    data_bare = DataSetofSets(datafull, datadrill_bare)

    # generate common variables
    sharesim = SharedSimulations(datadrill_bare)
    simMat = SimulationDraws(data_bare, M)

    # DataFrames for results
    df_t, df_Tstop = dataFramesForSimulations(simlist, datadrill_bare, Tstop)

    # send to workers
    set_g_BaseDataSetofSets(data_bare)
    set_g_SimulationDrawsMatrix(simMat)
    set_g_SharedSimulations(sharesim)

    for (k, (rwrd, wp, theta)) in enumerate(simlist)
        check_theta(datadrill, theta)
        fill!(sharesim, 0)
        simprim = set_g_SimulationPrimitives(rwrd, wp, Tstop, theta)
        pmap(i -> simulate_unit!(i,true), wrkrs, OneTo(N))
        # map(i -> simulate_unit!(simprim, i,true), OneTo(N))
        update_sim_dataframes_from_simdata(df_t, df_Tstop, sharesim, k)
    end

    return df_t, df_Tstop
end
