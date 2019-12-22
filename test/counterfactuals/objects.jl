# module ShaleDrillingLikelihood_CounterfactualObjects_Test
using Revise

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using Test

using ShaleDrillingLikelihood: drill, _x, _D, LeaseCounterfactual,
    ObservationGroup, InitialDrilling, ichars, zchars, uniti,
    end_ex0, _data, _i, simulate_lease!, Pprime,
    state_if_never_drilled, tstart, jtstart, first_state,
    update_sparse_state_transition!, vw_revenue,
    split_thetas, SimulationTmp, SimulationPrimitives,
    simulationPrimitives_information, dataFrames_from_simulationPrimitives,
    SharedSimulations, update_sim_dataframes_from_simdata!,
    value_function

num_i = 50
M = 10
nz = 13
nψ = 13
Tstop = 150

datafull, thetafull = last(MakeTestData(; num_i=num_i, nz=nz, nψ=nψ))

ddata = drill(datafull)
wpold = statespace(_model(ddata))

newrwrd = DrillReward(
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess(), NoLearn(), WithRoyalty()),
    DrillingCost_constant(),
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

@test ShaleDrillingLikelihood.num_i(datafull) ==
    ShaleDrillingLikelihood.num_i(data_for_xfer)

newdatafull = DataSetofSets(datafull, data_for_xfer)
theta = thetafull
sim = SimulationDraws(datafull, M)
update!(sim, ThetaRho())
sharesim = SharedSimulations(data_for_xfer)
simprim = SimulationPrimitives(newdatafull, sim, Tstop, theta, sharesim)
simtmp = SimulationTmp(simprim)

@assert _nparm(newrwrd) == _nparm(drill(newdatafull))

@test value_function(_model(drill(newdatafull))) == nothing
@test SimulationTmp(simprim) === simtmp

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
                obslast = 0
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

# to make sure state if not drilled is OK
@testset "iteration through state if never drilled in lease cntrfact" begin
    @test state_if_never_drilled(lc, 0) == 1

    @test tstart(lc) == firstindex(lc)
    @test length(lc) == length(jtstart(lc):length(zchars(ddata)))
    @test first_state(lc) == first(_x(l))
    @test _x(first(first(lc))) == _x(first(l))

    for (t, (obs,zt)) in enumerate(lc)
        if t == 1
            @test _x(obs) == first_state(lc)
        else
            _x(obs) > first_state(lc)
        end
    end
end # testset

@testset "trying out sparse_state_transition" begin
    thetasvw = split_thetas(newdatafull, theta)
    thet = first(thetasvw)
    for (obs,zt) in lc
        simm = getindex(view(sim, 1), rand(1:M))
        update_sparse_state_transition!(simtmp, obs, simm, thet)
    end
end


@testset "simulate 1 lease 1 time" begin
    simi = view(sim, uniti(l))
    simim = simi[1]

    fill!(sharesim, 0)
    simulate_lease!(simprim, l, simim, 1.0)

    @test abs(sum(sharesim.d0[:,1])) > 0
    @test abs(sum(sharesim.profit[:,1])) > 0
    @test abs(sum(sharesim.surplus[:,1])) > 0
    @test abs(sum(sharesim.revenue[:,1])) > 0
    @test abs(sum(sharesim.D_at_T[:,1])) > 0
end

using ShaleDrillingLikelihood: simulate_unit!

@testset "simulate unit" begin
    simprim2 = SimulationPrimitives(newdatafull, sim,
        newrwrd, wpold, Tstop, theta, sharesim)
        update!(sim, ThetaRho())
    fill!(sharesim, 0)
    do_r = false
    for i in 1:ShaleDrillingLikelihood.num_i(newdatafull)
        simulate_unit!(simprim2, i, do_r)
    end

    @test abs(sum(sharesim.d0        )) > 0
    @test abs(sum(sharesim.d1        )) > 0
    @test abs(sum(sharesim.d0psi     )) > 0
    @test abs(sum(sharesim.d1psi     )) > 0
    @test abs(sum(sharesim.d0eur     )) > 0
    @test abs(sum(sharesim.d1eur     )) > 0
    @test abs(sum(sharesim.d0eursq   )) > 0
    @test abs(sum(sharesim.d1eursq   )) > 0
    @test abs(sum(sharesim.d0eurcub  )) > 0
    @test abs(sum(sharesim.d1eurcub  )) > 0
    @test abs(sum(sharesim.epsdeq1   )) > 0
    @test abs(sum(sharesim.epsdgt1   )) > 0
    @test abs(sum(sharesim.Prdeq1    )) > 0
    @test abs(sum(sharesim.Prdgt1    )) > 0
    @test abs(sum(sharesim.Eeps      )) > 0
    @test abs(sum(sharesim.profit    )) > 0
    @test abs(sum(sharesim.surplus   )) > 0
    @test abs(sum(sharesim.revenue   )) > 0
    @test abs(sum(sharesim.drillcost )) > 0
    @test abs(sum(sharesim.extension )) > 0
    @test abs(sum(sharesim.D_at_T    )) > 0
end


@testset "using global objects" begin

    ddata = drill(datafull)
    wp = statespace(_model(ddata))
    rwrd = DrillReward(
        DrillingRevenue(Unconstrained(), NoTrend(), GathProcess(), NoLearn(), WithRoyalty()),
        DrillingCost_constant(),
        ExtensionCost_Constant()
    )

    ddm_novf = DDM_NoVF(_model(drill(datafull)))
    datadrill_dso = DataDrillStartOnly(drill(datafull))
    datadrill_bare= DataDrill(data_dso, ddm_novf)

    sharesim2 = SharedSimulations(datadrill_bare)
    fill!(sharesim2, 0)

    data_bare = DataSetofSets(datafull, datadrill_bare)
    simMat = SimulationDraws(data_bare, M)
    set_g_BaseDataSetofSets(data_bare)
    set_g_SimulationDrawsMatrix(simMat)
    set_g_SharedSimulations(sharesim2)

    simprim = set_g_SimulationPrimitives(rwrd, wp, Tstop, theta)

    @test simprim.sharedsim === sharesim2 #FIXME

    for i in 1:ShaleDrillingLikelihood.num_i(newdatafull)
        simulate_unit!(i, true)
    end

    @test all(isfinite.(sharesim2.d0))

    @test abs(sum(get_g_SharedSimulations().d0        )) > 0
    @test abs(sum(get_g_SharedSimulations().d1        )) > 0
    @test abs(sum(get_g_SharedSimulations().d0psi     )) > 0
    @test abs(sum(get_g_SharedSimulations().d1psi     )) > 0
    @test abs(sum(get_g_SharedSimulations().d0eur     )) > 0
    @test abs(sum(get_g_SharedSimulations().d1eur     )) > 0
    @test abs(sum(get_g_SharedSimulations().d0eursq   )) > 0
    @test abs(sum(get_g_SharedSimulations().d1eursq   )) > 0
    @test abs(sum(get_g_SharedSimulations().d0eurcub  )) > 0
    @test abs(sum(get_g_SharedSimulations().d1eurcub  )) > 0
    @test abs(sum(get_g_SharedSimulations().epsdeq1   )) > 0
    @test abs(sum(get_g_SharedSimulations().epsdgt1   )) > 0
    @test abs(sum(get_g_SharedSimulations().Prdeq1    )) > 0
    @test abs(sum(get_g_SharedSimulations().Prdgt1    )) > 0
    @test abs(sum(get_g_SharedSimulations().Eeps      )) > 0
    @test abs(sum(get_g_SharedSimulations().profit    )) > 0
    @test abs(sum(get_g_SharedSimulations().surplus   )) > 0
    @test abs(sum(get_g_SharedSimulations().revenue   )) > 0
    @test abs(sum(get_g_SharedSimulations().drillcost )) > 0
    @test abs(sum(get_g_SharedSimulations().extension )) > 0
    @test abs(sum(get_g_SharedSimulations().D_at_T    )) > 0

end



# end # module
