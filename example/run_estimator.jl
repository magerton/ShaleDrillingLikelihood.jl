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

M_cnstr = 50
M_full  = 50

do_cnstr = false
do_full  = true

maxtime_cnstr = 2 * 60^2
maxtime_full  = 3 * 60^2

REWARD = DrillReward(#
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess() ),
    # DrillingCost_TimeFE(2008,2012),
    DrillingCost_constant(),
    ExtensionCost_Constant()
)

ANTICIPATE = false
PSI = PsiSpace(21)
NUM_P = 21
NUM_C = 13
EXTEND_GRID = log(3)
MINP = minp_default()
DISCOUNT = RealDiscountRate()

DATADIR = "E:/projects/haynesville/intermediate_data"
# DATADIR = "/home/magerton/haynesville/intermediate_data"
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

# transition matrices
zrng, ztrans = GridTransition(data_drill_prim, EXTEND_GRID, NUM_P; minp=MINP)

# create drilling dataset
wp = statespace(data_drill_prim)
ddm = DynamicDrillModel(REWARD, DISCOUNT, wp, zrng, ztrans, PSI, ANTICIPATE)
data_drill = DataDrill(ddm, data_drill_prim)

# full dataset
dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))

# constrained model
rwrd_cnstr = ConstrainedProblem(REWARD, theta0_drill)
theta0_cnstr = ThetaConstrained(REWARD, theta0_drill)
data_cnstr = DataDrill(DynamicDrillModel(rwrd_cnstr, ddm), data_drill)
dataset_cnstr = DataSetofSets(data_cnstr, EmptyDataSet(), EmptyDataSet())

# ------------------- set up workers -----------------------

pids = start_up_workers(ENV)
@everywhere using ShaleDrillingLikelihood

# Solve constrained simpler model
if do_cnstr
    res_c, ew_c = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, maxtime_cnstr)
    updateThetaUnconstrained!(REWARD, theta0_drill, minimizer(res_c))
end

# Solve unconstrained full model
if do_full
    theta0s = (theta0_drill, theta0_royalty, theta0_produce)
    theta0_full = merge_thetas(theta0s, dataset_full)
    res_u, ew_u = solve_model(dataset_full, theta0_full, M_full, maxtime_full)
end
