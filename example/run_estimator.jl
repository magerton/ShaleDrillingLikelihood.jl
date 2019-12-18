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

M_cnstr = 250
M_full  = 250

DO_CNSTR = true
DO_FULL  = true
DO_NELDER = true

COMPUTE_INITIAL_VALUES = true

maxtime_cnstr = 1 * 60^2
maxtime_nelder = 1 * 60^2
maxtime_full  = 3 * 60^2

REWARD = DrillReward(#
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess() ),
    # DrillingCost_TimeFE(2008,2012),
    # DrillingCost_TimeFE_rigrate(2008,2012),
    DrillingCost_dgt1(),
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
if DO_CNSTR
    res_c, ew_c = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, maxtime_cnstr)
    theta1_cnstr = minimizer(res_c)
    updateThetaUnconstrained!(REWARD, theta0_drill, theta1_cnstr)
else
    # theta1_cnstr = [-0x1.aa7304a116f4ap+3, -0x1.2ba87c235a2f1p+3, -0x1.00759e1a5db57p+3, -0x1.d5c1f7b28424p+2, -0x1.ac628bffcfc0ep+2, 0x1.89909c23fb40ap+0, -0x1.e12af8551c9a1p+0, -0x1.6644de071c184p+1, 0x1.427b6a6fdeb06p-3, ]
end


# Solve unconstrained full model
if DO_FULL
    theta0s = (theta0_drill, theta0_royalty, theta0_produce)
    theta0_full = merge_thetas(theta0s, dataset_full)
    if DO_NELDER
        alg = ShaleDrillingLikelihood.nelder
        res_u_n, ew_u_n = solve_model(dataset_full, theta0_full, M_full, maxtime_nelder, alg)
        theta1_full = minimizer(res_u_n)
    else
        theta1_full = copy(theta0_full)
    end
    res_u, ew_u = solve_model(dataset_full, theta1_full, M_full, maxtime_full)
end
