# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    using ClusterManagers
end
using CountPlus, Distributed
using Optim: minimizer, Options, BFGS, NelderMead

# ------------------- number of simulations ----------------------

M_cnstr = 500
M_full  = 250

do_cnstr = false
do_full  = true

COMPUTE_INITIAL_VALUES = true

maxtime_cnstr = 6 * 60^2
maxtime_full  = 1* 60^2

REWARD = DrillReward(#
    DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() ),
    # DrillingCost_TimeFE_rigrate(2008,2012),
    DrillingCost_TimeFE(2008,2012),
    # DrillingCost_constant(),
    ExtensionCost_Constant()
)

ANTICIPATE = false
PSI = PsiSpace(15)
NUM_P = 15
NUM_C = 13
EXTEND_GRID = log(2)
MINP = minp_default()
DISCOUNT = RealDiscountRate()

if "SLURM_JOBID" in keys(ENV)
    DATADIR = "/home/magerton/haynesville/intermediate_data"
else
    DATADIR = "E:/projects/haynesville/intermediate_data"
end
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

# ------------------- get initial values -----------------------

if COMPUTE_INITIAL_VALUES
    roy_short = DataRoyalty(RoyaltyModelNoHet(), data_royalty)
    res_royalty = solve_model(roy_short, theta0_royalty[3:end])
    theta0_royalty[3:end], minimizer(res_royalty)

    theta0_produce .= ThetaProduceStarting(REWARD, rdatapath)
end

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
    # alg = ShaleDrillingLikelihood.nelder
    res_u, ew_u = solve_model(dataset_full, theta0_full, M_full, maxtime_full)
end
