# module ShaleDrillingLikelihood_OptimizeSpeedtests
using Revise

DOBTIME = false
DOPROFILE = false
DOPAR = true

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using BenchmarkTools
using Profile
using Distributed
using CountPlus
using Juno
using LinearAlgebra.BLAS: set_num_threads

using ShaleDrillingLikelihood: test_parallel_simloglik!

num_i = 15
M = 10

data_theta_sm, data_theta_lg = MakeTestData(;num_i=num_i)

nlogic_cores = length(Sys.cpu_info())
nphys_cores = Int(nlogic_cores/2)
stopcount!()

for nproc in (0, nphys_cores, nlogic_cores)

    println("added $nproc workers")

    rmprocs(workers())
    pids = addprocs(nproc; enable_threaded_blas=true)

    nth = Int(nlogic_cores / max(nproc,1))

    @eval @everywhere begin
        using LinearAlgebra.BLAS: set_num_threads
        set_num_threads($nth)
        using ShaleDrillingLikelihood
    end

    data, theta = data_theta_sm

    theta_peturb = 0.9 .* theta

    leo = LocalEstObj(data,theta_peturb)
    s = SimulationDraws(M, ShaleDrillingLikelihood.data(leo))
    reo = RemoteEstObj(leo, M)
    ew = EstimationWrapper(leo, reo)

    for dograd in false:true
        @eval @everywhere set_g_RemoteEstObj($reo)
        @btime test_parallel_simloglik!($ew, $theta_peturb, $dograd)
    end
end


# end
