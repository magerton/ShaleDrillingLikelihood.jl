using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using BenchmarkTools
using Profile
using Distributed
using CountPlus
using LinearAlgebra.BLAS: set_num_threads
using Base.Threads: nthreads

using ShaleDrillingLikelihood: test_parallel_simloglik!


num_i = 150
M = 100
nz = 51
nψ = 51

data_theta_sm, data_theta_lg = MakeTestData(; num_i=num_i, nz=nz, nψ=nψ)

nlogic_cores = length(Sys.cpu_info())
nphys_cores = Int(nlogic_cores/2)

rmprocs(filter(i -> i != 1, workers()))
nth = nthreads()
nproc = Int(nlogic_cores / nth)
pids = addprocs(nproc; enable_threaded_blas=true)

println("----------------------------------------------")
println("Using $nth threads and adding $nproc workers")
println("----------------------------------------------")

@everywhere begin
    using LinearAlgebra.BLAS: set_num_threads
    using Base.Threads: nthreads
    using ShaleDrillingLikelihood
    set_num_threads(nthreads())
end

data, theta = data_theta_sm

theta_peturb = 0.9 .* theta

leo = LocalEstObj(data,theta_peturb)
s = SimulationDraws(M, ShaleDrillingLikelihood.data(leo))
reo = RemoteEstObj(leo, M)
ew = EstimationWrapper(leo, reo)

stopcount!()
for dograd in false:true
    @eval @everywhere set_g_RemoteEstObj($reo)
    @btime test_parallel_simloglik!($ew, $theta_peturb, $dograd)
end


# end
