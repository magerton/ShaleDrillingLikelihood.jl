# module ShaleDrillingLikelihood_CounterfactualObjects_Test
using Revise

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using Test

using ShaleDrillingLikelihood: drill, _x, _D


num_i = 50
M = 10
nz = 13
nψ = 13
Tstop = 15

datafull = first(last(MakeTestData(; num_i=num_i, nz=nz, nψ=nψ)))

ddata = drill(datafull)
wpold = statespace(_model(ddata))


newrwrd = DrillReward(
    DrillingRevenue(Constrained(), TimeTrend(), GathProcess(), NoLearn(), WithRoyalty()),
    DrillingCost_TimeFE(2008,2016),
    ExtensionCost_Constant()
)
newwp = PerpetualProblem(wpold)


@testset "Revision of statespace" begin
    newd = DataDrill(ddata, newrwrd, newwp)
    for (xo, xn) in zip(_x(ddata), _x(newd))
        Do = _D(statespace(_model(ddata)), xo)
        Dn = _D(statespace(_model(newd)), xn)
        @test Do == Dn
    end
end

ddm_novf = DDM_NoVF(_model(ddata))
data_dso = DataDrillStartOnly(ddata)
data_for_xfer = DataDrill(data_dso, ddm_novf)

using ShaleDrillingLikelihood: SimulationTmp, SimulationPrimitives,
    simulationPrimitives_information, dataFrames_from_simulationPrimitives,
    SharedSimulations, update_sim_dataframes_from_simdata!

@which length(data_for_xfer)


ShaleDrillingLikelihood.num_i(datafull)
ShaleDrillingLikelihood.num_i(data_for_xfer)

newdatafull = DataSetofSets(datafull, data_for_xfer)
theta = rand(_nparm(newdatafull))
sim = SimulationDraws(datafull, M)
simtmp = SimulationTmp(data_for_xfer, M)
simprim = SimulationPrimitives(newdatafull, sim, Tstop, theta, simtmp)
sharesim = SharedSimulations(data_for_xfer)


simlist = [
    (newrwrd, PerpetualProblem(statespace(data_for_xfer)), theta),
]

df_t, df_Tstop = dataFrames_from_simulationPrimitives(simlist, data_for_xfer, Tstop)

update_sim_dataframes_from_simdata!(df_t, df_Tstop, 1, sharesim)

# end # module
