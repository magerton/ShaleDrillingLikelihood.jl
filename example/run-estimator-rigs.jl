# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    using ClusterManagers
end
using CountPlus, Distributed
using Optim: minimizer, Options, BFGS, NelderMead
using ShaleDrillingLikelihood: value_function, EVobj, cost
using SparseArrays: nonzeros
using Formatting: generate_formatter

num_nonzeros(x) = length(nonzeros(x))
intstr = generate_formatter("%'d")

# ------------------- number of simulations ----------------------

M_cnstr = 500
M_full  = 1000

DO_CNSTR = false
DO_FULL  = true

COMPUTE_INITIAL_VALUES = false

MAXTIME_CNSTR = 3 * 60^2
MAXTIME_FULL  = 48 * 60^2

THETA0_FULL_OVERRIDE = [-0x1.5744a54d60f62p+3, -0x1.d117aa07eef64p+2, -0x1.79f927f9bb9ffp+2, -0x1.441a10e192652p+2, -0x1.2acdb6e2bada1p+2, 0x1.9354425393b3cp+0, -0x1.782a92f1ee656p+0, -0x1.a3bab435dc7d7p+0, -0x1.4e03895da1d3ep+1, 0x1.316a8fef388e9p-1, 0x1.65c962aad6d6bp-2, 0x1.bcc7b91882082p-7, 0x1.527e753d41bep-1, 0x1.0c1cb074832d1p-3, 0x1.3137f3e64d182p-1, 0x1.2ffe4305080fap+0, -0x1.b6c5a5ac5414p+0, 0x1.1e815f092437cp-3, 0x1.e87168cddfdffp+1, 0x1.09aca495e0e38p+2, 0x1.405a56f44902cp+2, 0x1.7a0727a91f6d9p+2, 0x1.9eef181e16c0ap+2, -0x1.d8872e15169d5p+3, 0x1.8c727c253f856p-4, 0x1.481ac7a4734e9p-2, ]

COST = DrillingCost_TimeFE_rigrate(2008,2012)
EXT = ExtensionCost_Constant()
REV = DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() )
REWARD = DrillReward(REV, COST, EXT)
println("Model is\n\t$REWARD\n")


if cost(REWARD) isa DrillingCost_TimeFE
    PSI = PsiSpace(51)
    NUM_P = 51
    EXTEND_GRID = log(3)
    MINP = minp_default()
elseif cost(REWARD) isa DrillingCost_TimeFE_rigrate
    PSI = PsiSpace(13)
    NUM_P = 13
    EXTEND_GRID = log(2.5)
    MINP = 5e-5
end

ANTICIPATE = false
println("Firms anticipate T1EV shocks? $ANTICIPATE")

DISCOUNT = RealDiscountRate()
println("Psi-space is\n\t$PSI with length $(length(PSI))")

if "SLURM_JOBID" in keys(ENV)
    DATADIR = "/home/magerton/haynesville/intermediate_data"
else
    DATADIR = "E:/projects/haynesville/intermediate_data"
end
DATAPATH = "data_all_leases.RData"

# --------------- create data ---------------

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
println_time_flush("Z range is\n\t$zrng with lengths $(length.(zrng))")
println_time_flush("Transition matrix drops probs < $MINP and has $(intstr(num_nonzeros(ztrans))) elements")

# create drilling dataset
wp = statespace(data_drill_prim)
ddm = DynamicDrillModel(REWARD, DISCOUNT, wp, zrng, ztrans, PSI, ANTICIPATE)
data_drill = DataDrill(ddm, data_drill_prim)

evdims = size(EVobj(value_function(ddm)))
println_time_flush("EV dimension is $evdims, implying $(intstr(prod(evdims))) states")

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
    res_c, ew_c = solve_model(dataset_cnstr, theta0_cnstr, M_cnstr, MAXTIME_CNSTR)
    theta1_cnstr = minimizer(res_c)
    updateThetaUnconstrained!(REWARD, theta0_drill, theta1_cnstr)
end

# Solve unconstrained full model
theta0s = (theta0_drill, theta0_royalty, theta0_produce)
theta0_full = merge_thetas(theta0s, dataset_full)

if length(THETA0_FULL_OVERRIDE) == length(theta0_full)
    theta0_full .= THETA0_FULL_OVERRIDE
end

if DO_FULL
    println("Starting full model solution")
    res_u, ew_u = solve_model(dataset_full, theta0_full, M_full, MAXTIME_FULL)
end