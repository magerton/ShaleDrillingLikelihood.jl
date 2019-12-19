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
M_full  = 1000

DO_CNSTR = false # true
DO_NELDER = false
DO_FULL  = true

COMPUTE_INITIAL_VALUES = true

maxtime_cnstr = 1 * 60^2
maxtime_nelder = 1 * 60^2
maxtime_full  = 12 * 60^2

REWARD = DrillReward(#
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess() ),
    DrillingCost_TimeFE(2008,2012),
    # DrillingCost_TimeFE_rigrate(2008,2012),
    # DrillingCost_dgt1(),
    ExtensionCost_Constant()
)

ANTICIPATE = false
PSI = PsiSpace(31)
NUM_P = 51
NUM_C = 13
EXTEND_GRID = log(3)
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
println_time_flush("Data created")

# ------------------- get initial values -----------------------

if COMPUTE_INITIAL_VALUES
    roy_short = DataRoyalty(RoyaltyModelNoHet(), data_royalty)
    res_royalty = solve_model(roy_short, theta0_royalty[3:end])
    theta0_royalty[3:end], minimizer(res_royalty)

    theta0_produce .= ThetaProduceStarting(REWARD, rdatapath)
    println_time_flush("Pdxn / Royalty starting values done")
end

# ------------------- set up workers -----------------------

pids = start_up_workers(ENV)
@everywhere using ShaleDrillingLikelihood

# Solve constrained simpler model
if DO_CNSTR
    res_c, ew_c = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, maxtime_cnstr)
    theta1_cnstr = minimizer(res_c)
else
    theta1_cnstr = [-0x1.92e9287eaa0bp+3, -0x1.1f6b921d223ep+3, -0x1.ee1ec0acdb8e8p+2, -0x1.c76a3171bf78bp+2, -0x1.accd32a09433p+2, 0x1.9095a5a9b3f03p+0, -0x1.b1fe523465919p+0, -0x1.7332a38946ffcp+1, 0x1.51a64e52a723p-1, ]
end
updateThetaUnconstrained!(REWARD, theta0_drill, theta1_cnstr)


# Solve unconstrained full model
theta0s = (theta0_drill, theta0_royalty, theta0_produce)
theta0_full = merge_thetas(theta0s, dataset_full)
theta0_full .= [-0x1.9888015dc5e94p+3, -0x1.1eed252dfb0c2p+3, -0x1.e85a523a01567p+2, -0x1.c179cf4c98a4ep+2, -0x1.a0bb331b87f7p+2, 0x1.969dc0f15861ep+0, -0x1.88937177cf68cp+0, -0x1.4c5466cfc5e66p+1, 0x1.25aa7e67d4537p-1, 0x1.67f9929622178p-2, 0x1.581b47832f695p-2, 0x1.f2a9386c76f97p-4, 0x1.32db1364f88dfp-1, 0x1.2ed17e3a861c4p+0, -0x1.b3a5b2101c6bcp+0, 0x1.1e29f17d740f6p-3, 0x1.ee9af8421103cp+1, 0x1.0cbff947c745fp+2, 0x1.4365ea57b349bp+2, 0x1.7d05807825dfep+2, 0x1.a1defc1da86f4p+2, -0x1.d39937253c272p+3, 0x1.8c72785ac04dep-4, 0x1.47faa3985dcb7p-2, ]
if DO_FULL
    if DO_NELDER
        alg = ShaleDrillingLikelihood.nelder
        res_u_n, ew_u_n = solve_model(dataset_full, theta0_full, M_full, maxtime_nelder, alg)
        theta1_full = minimizer(res_u_n)
    else
        theta1_full = copy(theta0_full)
    end
    println("Starting full model solution")
    res_u, ew_u = solve_model(dataset_full, theta1_full, M_full, maxtime_full)
end
