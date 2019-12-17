using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

# detect if using SLURM
if "SLURM_JOBID" in keys(ENV)
    using ClusterManagers
end
using CountPlus, Optim, Test, Distributed

using DataFrames, CategoricalArrays, Dates, StatsModels, Query
using FileIO: load

using Optim: minimizer

# ------------------- number of simulations ----------------------

M_dcdp_dfree       = 175
M_dcdp_bfgs        = 250
M_full_dfree       = 175
M_full_bfgs        = 500

do_dcdp_dfree      = false
do_dcdp_bfgs       = true
do_full_dfree      = false
do_full_bfgs       = true

maxtime_dcdp_dfree = Int(   0.5 * 60^2)
maxtime_dcdp_bfgs  = Float64( 1 * 60^2)
maxtime_full_dfree = Int(     2 * 60^2)
maxtime_full_bfgs  = Float64(48 * 60^2)

REWARD = DrillReward(
    DrillingRevenue(Unconstrained(), NoTrend(), GathProcess() ),
    DrillingCost_TimeFE(2008,2012), # DrillingCost_constant(),
    ExtensionCost_Constant()
)

# # --------------- function -------------------
#
# function testsort(d::Dict, tbls::NTuple{N,String}, cols::NTuple{N,Symbol}) where {N}
#     for i = 1:N
#         tbl = tbls[i]
#         col = cols[i]
#         issorted(d[tbl][!,col]) || throw(error("table $tbl not sorted on $col!"))
#     end
#     println("tables are sorted")
# end
#
# --------------- load in data ---------------

datadir = "E:/projects/haynesville/intermediate_data"
datapath = "data_all_leases.RData"
rdatapath = joinpath(datadir, datapath)
data_rdata = load(rdatapath)


rwrd = REWARD
DataRoyalty(rwrd, rdatapath)
DataProduce(rwrd, rdatapath)
DataDrillPrimitive(rwrd, rdatapath)

# ---------------------------------------
#
# REWARD =  = DrillReward(
#     DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() ),
#     DrillingCost_TimeFE(2008,2012), # DrillingCost_constant(),
#     ExtensionCost_Constant()
# )
# ANTICIPATE_E = true
# CONSTR_COEFS = (log_ogip=0x1.47479e0efc894p-1, α_ψ=0x1.4388dea893293p-2, α_t=0x1.cf5fe057db0cfp-6, )
#
# Z_INFO = ShaleDrillingEstimation.PriceProcess.ProblemParms(;nlogp = 15, nlogc = 15, ngeology = 15, nψ = 15)
#
# println_time_flush("Data set")


# ------------------- set up workers -----------------------

rmprocs(getworkers())
pids = addprocs()
@everywhere using ShaleDrillingLikelihood

let theta=theta0, M=M, maxtime
    # estimation objects
    leo = LocalEstObj(d, theta)
    reo = RemoteEstObj(leo, M)
    ew = EstimationWrapper(leo, reo)
    leograd = ShaleDrillingLikelihood.grad(leo)

    resetcount!()
    startcount!([50, 500, 100000,], [1, 5, 100,])
    opts = Optim.Options(show_trace=true, time_limit=maxtime, allow_f_increases=true)

    res = solve_model(ew, theta; OptimOpts=opts)

    println(res)
    println("Recomputing final gradient / hessian")
    let dograd=true, theta=minimizer(res)
        parallel_simloglik!(ew, theta, dograd)
        update!(ew, theta, dograd)
    end
    println(coeftable(leo))
    print("Parameter estimates are\n\t")
    print(sprintf_binary(minimizer(res)))
    print("\n")
end
