# use https://jacobadenbaum.github.io/TexTables.jl
# use https://github.com/magerton/TexTables.jl
using ShaleDrillingLikelihood

TEX_TABLE_PATHS = "C:/Users/magerton/Desktop/test-table.tex"
MAIN_TEX_FILE = "./some-tables.tex"


FIRST_COST_YEAR = 2009
LAST_COST_YEAR = 2016


scriptdir = joinpath(dirname(pathof(ShaleDrillingLikelihood)), "..", "example")
include(joinpath(scriptdir, "display-coefs-fcts.jl"))

JLD2FILE = "E:/projects/ShaleDrillingResults/2019-12/16481962/estimation-results-16481962-no-rigs-data_all_leases.jld2"
rcnew = regcol("mytitle", nms_coef_se_sumstat(JLD2FILE)...)

mynote = "Mean well costs are the mean per-well drilling cost over the period $FIRST_COST_YEAR--$LAST_COST_YEAR. " *
         "This is calculated as \$\\frac{\\sigma_\\epsilon}{1-\\tau_{k}} \\frac{cost(2,s_{it},z_{it})}{2}\$ " *
         "where \$\\sigma_\\epsilon\$ is computed from \\eqref{eq:sigma-epsilon}. " *
         "The effective marginal corporate income tax is $(100*MGL_CORP_INC_TAX)\\%, " *
         "and the marginal tax rate on capital investment is \$\\tau_{k} \\approx $(round(100*CAPITAL_MGL_TAX_RATE;digits=1))\\% \$"


mt = make_table(regtable(rcnew, rcnew), "caption", "label", "note")
println(mt)



io = open(TEX_TABLE_PATHS, "w")
    write(io, "%!TEX root = " * MAIN_TEX_FILE * "\n\n\n\n")
    write(io, mt); write(io, "\n\n\n\n\n")
close(io)
