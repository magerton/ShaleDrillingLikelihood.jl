# use https://jacobadenbaum.github.io/TexTables.jl
# use https://github.com/magerton/TexTables.jl
using ShaleDrillingLikelihood

TEX_TABLE_PATHS = [
    "E:/projects/royalty-rates-and-drilling/writeup/JMPEdited/tables/results-2019-12.tex",
    "E:/projects/ShaleDrillingResults/results-2019-12.tex"
]
MAIN_TEX_FILE = "../JMPEdited.tex"

FIRST_COST_YEAR = 2009
LAST_COST_YEAR = 2016

scriptdir = joinpath(dirname(pathof(ShaleDrillingLikelihood)), "..", "example")
include(joinpath(scriptdir, "display-coefs-fcts.jl"))

JLD2FILE = "E:/projects/ShaleDrillingResults/2019-12/16481962/estimation-results-16481962-no-rigs-data_all_leases.jld2"


title_dataset = [
     ("All"          , "E:/projects/ShaleDrillingResults/2019-12/16481962/estimation-results-16481962-no-rigs-data_all_leases.jld2",                  ),
     ("First"        , "E:/projects/ShaleDrillingResults/2019-12/16627219/estimation-results-16627219-no-rigs-data_first_lease_only.jld2",            ),
     ("First, restr" , "E:/projects/ShaleDrillingResults/2019-12/16482515/estimation-results-16482515-no-rigs-data_first_lease_only_all_leased.jld2", ),
     ("Last"         , "E:/projects/ShaleDrillingResults/2019-12/16482550/estimation-results-16482550-no-rigs-data_last_lease_only.jld2",             ),
     ("All, rigs"    , "E:/projects/ShaleDrillingResults/2019-12/16486124/estimation-results-16486124-WITH-rigs-data_all_leases.jld2",                ),
     ("T1EV"         , "E:/projects/ShaleDrillingResults/2019-12/16627762/estimation-results-16627762-no-rigs-data_all_leases.jld2",                  ),
 ]

rcs = [regcol(t, nms_coef_se_sumstat(j)...) for (t,j) in title_dataset]


# idx_rho = 12
# idxs = vcat(13:21,1:12,22:25)
# thetarho = coef[idx_rho]
# se[idx_rho] *= _dρdθρ(thetarho)
# coef[idx_rho] = _ρ(thetarho)
# coef[idxs], se[idxs]

mynote = "Mean well costs are the mean per-well drilling cost over the period $FIRST_COST_YEAR--$LAST_COST_YEAR. " *
    "This is calculated as \$\\frac{\\sigma_\\epsilon}{1-\\tau_{k}} \\frac{cost(2,s_{it},z_{it})}{2}\$ " *
    "where \$\\sigma_\\epsilon\$ is computed from \\eqref{eq:sigma-epsilon}. " *
    "The effective marginal corporate income tax is $(100*MGL_CORP_INC_TAX)\\%, " *
    "and the marginal tax rate on capital investment is \$\\tau_{k} \\approx $(round(100*CAPITAL_MGL_TAX_RATE;digits=1))\\% \$"

mt = make_table(regtable(rcs...), "Estimates for full model", "tab:all-estimates", mynote)
println(mt)

for t in TEX_TABLE_PATHS
    io = open(t, "w")
        write(io, "%!TEX root = " * MAIN_TEX_FILE * "\n\n\n\n")
        write(io, mt); write(io, "\n\n\n\n\n")
    close(io)
end
