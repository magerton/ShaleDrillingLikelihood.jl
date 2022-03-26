module SDLParameters

using Distributions
using ShaleDrillingLikelihood
using Dates
using SparseArrays
using LinearAlgebra
using Random

using StatsModels: @formula
using CategoricalArrays: CategoricalVector
using MixedModels

using ShaleDrillingLikelihood: simulate,
    cost, extend, revenue, scrap,
    lrmean, lrvar, lrstd,
    zero_out_small_probs,
    extra_parm,
    LeasedProblem,
    idx_produce_ψ,
    idx_drill_ψ,
    idx_produce_g,
    idx_drill_g,
    idx_produce_t,
    idx_drill_t,
    idx_drill_D,
    idx_produce_D,
    total_wells_drilled,
    DataDynamicDrill,
    num_choices,
    _num_x,
    DrillingRevenueUnconstrained

using Base: product
using LinearAlgebra: checksquare
using ArgParse
using Base.Meta: parse
import ArgParse: parse_item # SDL

import ShaleDrillingLikelihood: ichars, zchars,
    DataRoyalty, DataProduce, DataDrillPrimitive, DataDrill

include("thetas.jl")           # starting values
include("model-components.jl") # components to simulate things
include("make-models.jl")      # generate test data
include("grids.jl")            # price, cost, year grids
include("data-import.jl")      # generate problem data from RData
include("argparse.jl")         # for solving model
include("argparse-simulations.jl") # for counterfactuals

end
