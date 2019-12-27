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
using ShaleDrillingLikelihood: value_function, EVobj, cost
using SparseArrays: nonzeros
using Formatting: generate_formatter
using Dates: today
using RCall

num_nonzeros(x) = length(nonzeros(x))
intstr = generate_formatter("%'d")

using ShaleDrillingLikelihood: learn, royalty, constr, tech, tax,
    PerfectInfo, MaxLearning, drill,
    simulationPrimitives_information,
    doSimulations, average_cost_df,
    Theta_NoTech, theta_revenue,
    time_idx, DateQuarter

# ------------------- number of simulations ----------------------

pargs = parse_commandline_counterfactuals()
print_parsed_args_counterfactuals(pargs)

DO_PAR         = !pargs["noPar"]
JLD2FILE       = pargs["jld2"]
DATE_STOP      = pargs["dateStop"]
TECH_YEAR_ZERO = pargs["techYearZero"]
RFILEDIR       = pargs["rFileDir"]

# --------------- create data ---------------

if "SLURM_JOBID" in keys(ENV)
    DATADIR = "/home/magerton/haynesville/intermediate_data"
else
    DATADIR = "E:/projects/haynesville/intermediate_data"
end

println_time_flush("Loading results from $JLD2FILE")
file = jldopen(JLD2FILE, "r")
    DATAPATH = file["DATAPATH"]
    M        = file["M"]
    ddmnovf  = file["ddm_novf"]
    theta    = file["theta1"]
close(file)
println_time_flush("Done!")

REWARD = reward(ddmnovf)
ddm = DynamicDrillModel(ddmnovf, REWARD)

# --------------- create data ---------------

# load in data from disk
rdatapath = joinpath(DATADIR, DATAPATH)
println_time_flush("loading $rdatapath")

data_royalty    = DataRoyalty(REWARD, rdatapath)
data_produce    = DataProduce(REWARD, rdatapath)
data_drill_prim = DataDrillPrimitive(REWARD, rdatapath)
data_drill      = DataDrill(data_drill_prim, ddm)

evdims = size(EVobj(value_function(ddm)))
println_time_flush("EV dimension is $evdims, implying $(intstr(prod(evdims))) states")

# full dataset
dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))

println_time_flush("Data created")

let thts = split_thetas(dataset_full, theta)
    println("theta_drill = $(round.(thts[1]; digits=3))")
    println("theta_roy   = $(round.(thts[2]; digits=3))")
    println("theta_pdxn  = $(round.(thts[3]; digits=3))")
end

# ------------------- simulations -----------------------

R = ShaleDrillingLikelihood.revenue(REWARD)
C = ShaleDrillingLikelihood.cost(REWARD)
E = ShaleDrillingLikelihood.extend(REWARD)

wp = statespace(ddm)
PP = PerpetualProblem(wp)

# ------------------- average costs -----------------------

cost_df = average_cost_df(dataset_full, theta)

# ------------------- simulation info -----------------------

DrillRev(lrn, roy) = DrillingRevenue(constr(R), tech(R), tax(R), lrn, roy)
DrillRwrd(lrn, roy) = DrillReward(DrillRev(lrn,roy), C, E)

theta_notech = Theta_NoTech(dataset_full, theta, TECH_YEAR_ZERO)
TSTOP = time_idx(zchars(drill(dataset_full)), DateQuarter(DATE_STOP))

simlist = [
    (DrillRwrd( learn(R),      royalty(R) ),   wp, theta, ),           # "Baseline"
    (DrillRwrd( learn(R),      royalty(R) ),   wp, theta_notech, ),    # "Where to drill"
    (DrillRwrd( NoLearn(),     royalty(R) ),   wp, theta, ),           # "No update ($\\psi_{it} = \\psi_i^0$)"
    (DrillRwrd( PerfectInfo(), royalty(R) ),   wp, theta, ),           # "Perfect information ($\\psi_{it} = \\psi_i^1$)"
    (DrillRwrd( MaxLearning(), royalty(R) ),   wp, theta, ),           # "Uninformative signals ($\\rho = 0$)"
    (DrillRwrd( learn(R),      NoRoyalty()),   wp, theta, ),           # "No royalty"
    (DrillRwrd( learn(R),      royalty(R) ),   PP, theta, ),           # "No expiration"
    (DrillRwrd( learn(R),      NoRoyalty()),   PP, theta, ),           # "Ownership"
    (DrillRwrd(NoLearn(),      NoRoyalty()  ), PP, theta_notech, ),    # "Price only"
    (DrillRwrd(NoLearn(),      NoRoyalty()  ), PP, theta, ),           # "Price + tech (no distortions)"
    (DrillRwrd(NoLearn(),      WithRoyalty()), wp, theta_notech, ),    # "Price + leases"
    (DrillRwrd(Learn(),        NoRoyalty()  ), PP, theta_notech, ),    # "Price + learning"
    (DrillRwrd(NoLearn(),      WithRoyalty()), PP, theta, ),           # "How to drill"
]

simlist_meta = [(a,b,copy(theta_revenue(dataset_full, c))) for (a,b,c) in simlist]

simulations_meta_info = simulationPrimitives_information(simlist_meta)

# ------------------- simulations -----------------------

if DO_PAR
    pids = start_up_workers(ENV)
    @everywhere using ShaleDrillingLikelihood
    println_time_flush("Library loaded on workers")
end

df_d, df_D = doSimulations(dataset_full, simlist, TSTOP, M)

# ------------------- save -----------------------

dpath_no_rdata = replace(DATAPATH, ".RData" => "")
filenm = "simulations-$(SLURM_JOBID)-$(dpath_no_rdata)-$(today()).RData"
filepath = joinpath(RFILEDIR, filenm)

println_time_flush("Saving simulations to $filepath")

THETA = theta
FILEPATH = filepath

@rput df_d df_D cost_df simulations_meta_info THETA FILEPATH TSTOP

R"""
library(data.table)
setDT(df_D)
setDT(df_d)
setDT(cost_df)
setDT(simulations_meta_info)

save(simulations_meta_info, df_d, df_D, cost_df, THETA, TSTOP, file=FILEPATH)
rm(  simulations_meta_info, df_d, df_D, cost_df, THETA, TSTOP,      FILEPATH)
gc()
"""


println_time_flush("Mischief managed. :)")
