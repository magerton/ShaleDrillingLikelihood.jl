using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using SharedArrays
using JLD2
using RCall
using DataFrames
using Dates: today

using ShaleDrillingLikelihood: _ψ2, royalty, produce, num_i, value_function, EVobj, cost, drill
using Base.Iterators: flatten, product, OneTo

using Formatting: generate_formatter
num_nonzeros(x) = length(nonzeros(x))
intstr = generate_formatter("%'d")


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
    SLURM_JOBID = ENV["SLURM_JOBID"]
else
    DATADIR = "E:/projects/haynesville/intermediate_data"
    SLURM_JOBID = "999999"
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

# ------------------- parallel startup -----------------------

if DO_PAR
    pids = start_up_workers(ENV)
    @everywhere using ShaleDrillingLikelihood
    println_time_flush("Library loaded on workers")
end

# ------------------- simulations -----------------------

N = num_i(dataset_full)
posteriors = SharedPosterior(M, N; pids = pids)
sim = SimulationDraws(M, dataset_full)

@eval @everywhere set_g_BaseDataSetofSets($dataset_full)
@eval @everywhere set_g_SimulationDrawsMatrix($sim)
@eval @everywhere set_g_SharedPosterior($posteriors)
@eval @everywhere ShaleDrillingLikelihood.update!(get_g_BaseDataSetofSets(), $theta)
println_time_flush("starting simulations")

# map(i -> simloglik_posterior!(i, theta, sim, posteriors, dataset_full), OneTo(N))
pmap(i -> simloglik_posterior!(i, theta), CachingPool(pids), OneTo(N))
rmprocs(pids)

unitidx = [Int32(n) for (m,n) in product(OneTo(M), OneTo(N))]
psi2    = _ψ2(sim)
post_d  = sdata(drill(posteriors))
post_r  = sdata(royalty(posteriors))
post_p  = sdata(produce(posteriors))

posterior_df = DataFrame(
    uniti      = vec(unitidx),
    psi2       = vec(psi2),
    post_drill = vec(post_d),
    post_roy   = vec(post_r),
    post_pdxn  = vec(post_p)
)

# ------------------- save -----------------------

dpath_no_rdata = replace(DATAPATH, ".RData" => "")
filenm = "posterior-$(SLURM_JOBID)-$(dpath_no_rdata)-$(today()).RData"
FILEPATH = joinpath(RFILEDIR, filenm)

println("saving posterior to $FILEPATH")
flush(stdout)

@rput posterior_df FILEPATH

R"""
library(data.table)
setDT(posterior_df)
save(posterior_df, file=FILEPATH)
rm(posterior_df, FILEPATH)
gc()
"""

println("all done! :)")
flush(stdout)
