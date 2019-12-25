# module ShaleDrillingLikelihood_CounterfactualObjects_Test
using Revise

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using Test
using SharedArrays

using ShaleDrillingLikelihood: drill, _x, _D, LeaseCounterfactual,
    ObservationGroup, InitialDrilling, ichars, zchars, uniti,
    end_ex0, _data, _i, simulate_lease!, Pprime,
    state_if_never_drilled, tstart, jtstart, first_state,
    update_sparse_state_transition!, vw_revenue,
    split_thetas, SimulationTmp, SimulationPrimitives,
    simulationPrimitives_information, dataFramesForSimulations,
    SharedSimulations, update_sim_dataframes_from_simdata!,
    value_function,check_theta, doSimulations

num_i = 50
M = 10
nz = 13
nψ = 13

datafull, thetafull = last(MakeTestData(; num_i=num_i, nz=nz, nψ=nψ))

using ShaleDrillingLikelihood: SharedPosterior, _qm, _ψ2, royalty, produce,
    num_i

posteriors = SharedPosterior(datafull, M)
sim = SimulationDraws(M, datafull)

@eval @everywhere set_g_BaseDataSetofSets($datafull)
@eval @everywhere set_g_SimulationDrawsMatrix($sim)
@eval @everywhere set_g_SharedPosterior($posteriors)

N = num_i(datafull)

let x=theta, dograd=dograd, kwargs=kwargs
    pmap(i -> simloglik_posterior!(i, x), wp, OneTo(N))
end


psi2 = _ψ2(sim)
post_d = sdata(drill(posteriors))
post_r = sdata(royalty(posteriors))
post_p = sdata(produce(posteriors))
