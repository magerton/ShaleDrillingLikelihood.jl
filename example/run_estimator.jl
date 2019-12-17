# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    using ClusterManagers
end
using CountPlus, Distributed
using Optim: minimizer, Options

# ------------------- number of simulations ----------------------

M_cnstr = 250
M_full  = 500

do_cnstr = true
do_full  = true

maxtime_cnstr = 1 * 60^2
maxtime_full  = 1 * 60^2

REWARD = DrillReward(
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess() ),
    DrillingCost_TimeFE(2008,2012), # DrillingCost_constant(),
    ExtensionCost_Constant()
)

ANTICIPATE = true
PSI = PsiSpace(21)
NUM_P = 21
NUM_C = 13
EXTEND_GRID = log(3)
MINP = minp_default()
DISCOUNT = RealDiscountRate()

# DATADIR = "D:/projects/haynesville/intermediate_data"
DATADIR = "/home/magerton/haynesville/intermediate_data"
DATAPATH = "data_all_leases.RData"

# --------------- create data ---------------

# constrained version of reward function

# load in data from disk
rdatapath = joinpath(DATADIR, DATAPATH)
data_royalty    = DataRoyalty(REWARD, rdatapath)
data_produce    = DataProduce(REWARD, rdatapath)
data_drill_prim = DataDrillPrimitive(REWARD, rdatapath)

thetarho0 = 1.178623255316409
theta0_royalty = Theta(data_royalty; θρ = thetarho0)
theta0_produce = Theta(data_produce)
theta0_drill   = Theta(REWARD; θρ=thetarho0)
theta0s = (theta0_drill, theta0_royalty, theta0_produce)

# transition matrices
zrng, ztrans = GridTransition(data_drill_prim, EXTEND_GRID, NUM_P; minp=MINP)

# create drilling dataset
wp = statespace(data_drill_prim)
ddm = DynamicDrillModel(REWARD, DISCOUNT, wp, zrng, ztrans, PSI, ANTICIPATE)
data_drill = DataDrill(ddm, data_drill_prim)

# full dataset
dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))
theta0_full = merge_thetas(theta0s, dataset_full)

# constrained model
rwrd_cnstr = ConstrainedProblem(REWARD, theta0_drill)
theta0_cnstr = ThetaConstrained(REWARD, theta0_drill)
data_cnstr = DataDrill(DynamicDrillModel(rwrd_cnstr, ddm), data_drill)
dataset_cnstr = DataSetofSets(data_cnstr, EmptyDataSet(), EmptyDataSet())

# ------------------- set up workers -----------------------

pids = start_up_workers(ENV)
@everywhere using ShaleDrillingLikelihood

res, ew = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, maxtime_cnstr)

# let d=dataset_cnstr, theta=theta0_cnstr, M=M_cnstr, maxtime=maxtime_cnstr
#     # estimation objects
#     leo = LocalEstObj(d, theta)
#     reo = RemoteEstObj(leo, M)
#     ew = EstimationWrapper(leo, reo)
#     leograd = ShaleDrillingLikelihood.grad(leo)
#     @eval @everywhere set_g_RemoteEstObj($reo)
#
#     resetcount!()
#     startcount!([100, 500, 100000,], [1, 5, 100,])
#     opts = Optim.Options(show_trace=true, time_limit=maxtime, allow_f_increases=true)
#
#     res = solve_model(ew, theta; OptimOpts=opts)
#
#     # println(res)
#     println("Recomputing final gradient / hessian")
#     let dograd=true, theta=minimizer(res)
#         parallel_simloglik!(ew, theta, dograd)
#         update!(ew, theta, dograd)
#     end
#     println(coeftable(leo))
#     print("Parameter estimates are\n\t")
#     print(sprintf_binary(minimizer(res)))
#     print("\n")
# end
