# module ShaleDrillingLikelihood_CounterfactualObjects_Test
using Revise

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using Test

using ShaleDrillingLikelihood: drill, _x, _D, LeaseCounterfactual,
    ObservationGroup, InitialDrilling, ichars, zchars, uniti,
    end_ex0, _data, _i, simulate_lease!, Pprime


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
update!(sim, ThetaRho())
sharesim = SharedSimulations(data_for_xfer)
simprim = SimulationPrimitives(newdatafull, sim, Tstop, theta, sharesim)
simtmp = SimulationTmp(simprim)


simlist = [
    (newrwrd, PerpetualProblem(statespace(data_for_xfer)), theta),
]

df_t, df_Tstop = dataFrames_from_simulationPrimitives(simlist, data_for_xfer, Tstop)

update_sim_dataframes_from_simdata!(df_t, df_Tstop, 1, sharesim)

# make these objects
u = ObservationGroup(ddata, 1)
r = ObservationGroup(u, InitialDrilling())
l = ObservationGroup(r, 1)
lc = LeaseCounterfactual(l)

@testset "last counterfactual is an undrilled state" begin
    for unit in ddata
        for regime in unit
            for lease in regime
                lc = LeaseCounterfactual(lease)
                @test length(lc) >= 0
                for (t,(obs,zt)) in enumerate(lc)
                    if t == length(lc)
                        wp = statespace(_model(obs))
                        @test _x(obs) == end_ex0(wp)+1
                    end
                end
            end
        end
    end
end

using ShaleDrillingLikelihood: update_sparse_state_transition!, vw_revenue, split_thetas

@testset "trying out sparse_state_transition" begin
    thetasvw = split_thetas(newdatafull, theta)
    thet = first(thetasvw)
    for (obs,zt) in lc
        simm = getindex(view(sim, 1), rand(1:M))
        update_sparse_state_transition!(simtmp, obs, simm, thet)
    end
end



@test SimulationTmp(simprim) === simtmp
#
# ShaleDrillingLikelihood.reset!(SimulationTmp(simprim), 3)
# thetasvw = split_thetas(newdatafull, theta)
# thet = first(thetasvw)
#
#
# ShaleDrillingLikelihood.update!(simtmp, first(first(lc)), simm, thet)
#
# simtmp.Pprime[:,2]

# _x(l)
#
# simtmp.sa
#@test sum(sharesim.d0        ) > 0
@testset "simulate 1 lease" begin
    @test sum(sharesim.d1        ) > 0
    @test sum(sharesim.d0psi     ) > 0
    @test sum(sharesim.d1psi     ) > 0
    @test sum(sharesim.d0eur     ) > 0
    @test sum(sharesim.d1eur     ) > 0
    @test sum(sharesim.d0eursq   ) > 0
    @test sum(sharesim.d1eursq   ) > 0
    @test sum(sharesim.d0eurcub  ) > 0
    @test sum(sharesim.d1eurcub  ) > 0
    @test sum(sharesim.epsdeq1   ) > 0
    @test sum(sharesim.epsdgt1   ) > 0
    @test sum(sharesim.Prdeq1    ) > 0
    @test sum(sharesim.Prdgt1    ) > 0
    @test sum(sharesim.Eeps      ) > 0
    @test sum(sharesim.profit    ) > 0
    @test sum(sharesim.surplus   ) > 0
    @test sum(sharesim.revenue   ) > 0
    @test sum(sharesim.drillcost ) > 0
    @test sum(sharesim.extension ) > 0
end

@test sum(sharesim.D_at_T) >0

# end

# end # module
