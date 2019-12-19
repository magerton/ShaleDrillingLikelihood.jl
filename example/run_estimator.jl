# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    using ClusterManagers
end
using CountPlus, Distributed
using Optim: minimizer, Options, BFGS, NelderMead
using ShaleDrillingLikelihood: value_function, EVobj

# ------------------- number of simulations ----------------------

M_cnstr = 500
M_full  = 1000

DO_CNSTR = true # true
DO_FULL  = true

COMPUTE_INITIAL_VALUES = true

maxtime_cnstr = 3 * 60^2
maxtime_nelder = 1 * 60^2
maxtime_full  = 8 * 60^2

REWARD = DrillReward(
    DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() ),
    # DrillingCost_TimeFE(2008,2012),
    DrillingCost_TimeFE_rigrate(2008,2012),
    # DrillingCost_dgt1(),
    ExtensionCost_Constant()
)
println("Model is\n")
println(REWARD)
println("")

ANTICIPATE = false
PSI = PsiSpace(13)
NUM_P = 13
NUM_C = 13
EXTEND_GRID = log(2.5)
MINP = minp_default()
DISCOUNT = RealDiscountRate()
println("Psi-space is\n\t$PSI")

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
println_time_flush("Data imported")

thetarho0 = ThetaRho()
theta0_royalty = Theta(data_royalty; θρ = thetarho0)
theta0_produce = Theta(data_produce)
theta0_drill   = Theta(REWARD; θρ=thetarho0)

# transition matrices
zrng, ztrans = GridTransition(data_drill_prim, EXTEND_GRID, NUM_P; minp=MINP)
println_time_flush("Z range is\n\t$zrng")

# create drilling dataset
wp = statespace(data_drill_prim)
ddm = DynamicDrillModel(REWARD, DISCOUNT, wp, zrng, ztrans, PSI, ANTICIPATE)
data_drill = DataDrill(ddm, data_drill_prim)

evdims = size(EVobj(value_function(ddm)))
println_time_flush("EV dimension is $evdims, implying $(prod(evdims)) states")

# full dataset
dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))

# constrained model
rwrd_cnstr = ConstrainedProblem(REWARD, theta0_drill)
theta0_cnstr = ThetaConstrained(REWARD, theta0_drill)
data_cnstr = DataDrill(DynamicDrillModel(rwrd_cnstr, ddm), data_drill)
dataset_cnstr = DataSetofSets(data_cnstr, EmptyDataSet(), EmptyDataSet())
println_time_flush("Data created")

# ------------------- get initial values -----------------------

if COMPUTE_INITIAL_VALUES
    roy_short = DataRoyalty(RoyaltyModelNoHet(), data_royalty)
    res_royalty = solve_model(roy_short, theta0_royalty[3:end])
    theta0_royalty[3:end] .= minimizer(res_royalty)

    theta0_produce .= ThetaProduceStarting(REWARD, rdatapath)
    println_time_flush("Pdxn / Royalty starting values done")
end

# ------------------- set up workers -----------------------

pids = start_up_workers(ENV)
@everywhere using ShaleDrillingLikelihood
println_time_flush("Library loaded on workers")

# Solve constrained simpler model
if DO_CNSTR
    res_c, ew_c = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, maxtime_cnstr)
    theta1_cnstr = minimizer(res_c)
    updateThetaUnconstrained!(REWARD, theta0_drill, theta1_cnstr)
end

# Solve unconstrained full model
theta0s = (theta0_drill, theta0_royalty, theta0_produce)
theta0_full = merge_thetas(theta0s, dataset_full)

if DO_FULL
    println("Starting full model solution")
    res_u, ew_u = solve_model(dataset_full, theta0_full, M_full, maxtime_full)
end
