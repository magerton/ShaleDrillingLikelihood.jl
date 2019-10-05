module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra
using Base.Threads

abstract type AbstractIntermediateComputations end


include("data-access.jl")
include("sum-functions.jl")

include("flow.jl")

include("models.jl")

include("royalty.jl")
include("production.jl")
include("drilling.jl")

# include("overall-likelihood.jl")

end # module
