# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    SLURM_JOBID = ENV["SLURM_JOBID"]
    using ClusterManagers
else
    SLURM_JOBID = ""
end

using CountPlus, Distributed, JLD2
using Optim: minimizer, Options, BFGS, NelderMead
using ShaleDrillingLikelihood: value_function, EVobj, cost, serial_simloglik!
using SparseArrays: nonzeros
using Formatting: generate_formatter

num_nonzeros(x) = length(nonzeros(x))
intstr = generate_formatter("%'d")

# ------------------- number of simulations ----------------------

pargs = parse_commandline()
# using ArgParse: parse_args
# pargs = parse_args(["--noPar"], ShaleDrillingLikelihood.SDLParameters.arg_settings())
print_parsed_args(pargs)

DATAPATH = pargs["dataset"]

# estimation
M_cnstr = pargs["Mcnstr"]
M_full  = pargs["Mfull"]
DO_CNSTR = pargs["doCnstr"]
DO_FULL  = !pargs["noFull"]
MAXTIME_CNSTR = pargs["maxtimeCnstr"]
MAXTIME_FULL  = pargs["maxtimeFull"]
DO_PAR = !pargs["noPar"]

# parms
COMPUTE_INITIAL_VALUES = pargs["computeStarting"]
THETA0_FULL_OVERRIDE = pargs["theta"]

#payoffs
COST = pargs["cost"]
EXT = pargs["extension"]
REV = pargs["revenue"]
REWARD = DrillReward(REV, COST, EXT)
ANTICIPATE = pargs["anticipateT1EV"]
DISCOUNT = pargs["discount"]

# approximation
PSI = PsiSpace(pargs["numPsi"])
NUM_P = pargs["numP"]
EXTEND_GRID = pargs["extendPriceGrid"]
MINP = pargs["minTransProb"]

# print some things
println("Model is\n\t$REWARD\n")
println("Firms anticipate T1EV shocks? $ANTICIPATE")
println("Psi-space is\n\t$PSI with length $(length(PSI))")

if cost(REWARD) isa DrillingCost_TimeFE
    HASRIGS = "no-rigs"
elseif cost(REWARD) isa DrillingCost_TimeFE_rigrate
    HASRIGS = "WITH-rigs"
elseif cost(REWARD) isa DrillingCost_TimeFE_costdiffs
    HASRIGS = "COSTDIFF"
else
    throw(error("don't have values for this cost fct"))
end

if "SLURM_JOBID" in keys(ENV)
    DATADIR = "/home/magerton/haynesville/intermediate_data"
else
    DATADIR = "E:/projects/haynesville/intermediate_data"
end

# --------------- create data ---------------

# load in data from disk
rdatapath = joinpath(DATADIR, DATAPATH)
println_time_flush("loading $rdatapath")
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
data_drill = DataDrill(data_drill_prim, ddm)

evdims = size(EVobj(value_function(ddm)))
println_time_flush("EV dimension is $evdims, implying $(intstr(prod(evdims))) states")

# full dataset
dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))

# Solve unconstrained full model
theta0s = (theta0_drill, theta0_royalty, theta0_produce)
theta0_full = merge_thetas(theta0s, dataset_full)

leo = LocalEstObj(dataset_full, theta0_full)
reo = RemoteEstObj(leo, M_full)
ew = EstimationWrapper(leo, reo)

dograd = true

serial_simloglik!(ew, theta0_full, dograd; n=1)
