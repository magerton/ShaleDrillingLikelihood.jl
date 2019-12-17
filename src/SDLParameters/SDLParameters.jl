module SDLParameters

using Distributions
using ShaleDrillingLikelihood
using Dates
using SparseArrays
using LinearAlgebra
using Random

using StatsModels: @formula

using ShaleDrillingLikelihood: simulate,
    cost, extend, revenue,
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
    total_wells_drilled,
    DataDynamicDrill

using Base: product
using LinearAlgebra: checksquare

import ShaleDrillingLikelihood: ichars

include("thetas.jl")
include("model-components.jl")
include("make-models.jl")
include("grids.jl")
include("data-import.jl")

end
