using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

using Profile
using CountPlus
using Juno
using BenchmarkTools

using ShaleDrillingLikelihood: test_serial_simloglik!

num_i = 15
M = 60
nz = 101
nψ = 101

data_theta_sm, data_theta_lg = MakeTestData(; num_i=num_i, nz=nz, nψ=nψ)

stopcount!()

data, theta = data_theta_lg

theta_peturb = 0.9 .* theta
leo = LocalEstObj(data,theta_peturb)
s = SimulationDraws(M, ShaleDrillingLikelihood.data(leo))
reo = RemoteEstObj(leo, M)
ew = EstimationWrapper(leo, reo)


# run a couple of times to compile
VFTOL = 1e-9
for i = 1:2
    test_serial_simloglik!(ew, theta_peturb, true; vftol=VFTOL)
    test_serial_simloglik!(ew, theta_peturb, true; vftol=VFTOL)
end

# profile
@profile test_serial_simloglik!(ew, theta_peturb, true; vftol=VFTOL)
Juno.profiletree()
Juno.profiler()

# different VF Tols
for VFTOL in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
    println("VFTOL = $VFTOL")
    @btime test_serial_simloglik!($ew, $theta_peturb, true; vftol=$VFTOL)
end
