module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra

abstract type AbstractIntermediateComputations end

include("tmp.jl")
include("models.jl")
include("royalty.jl")
# include("production.jl")
# include("drilling.jl")

end # module
