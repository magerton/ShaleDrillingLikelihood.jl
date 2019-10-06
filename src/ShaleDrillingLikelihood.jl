module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra
using Base.Threads
using Halton

abstract type AbstractIntermediateComputations end
abstract type AbstractTempVar end

const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}


include("models.jl")

include("simulation-draws.jl")
include("data-structure.jl")
include("data-access.jl")
include("sum-functions.jl")

include("flow.jl")


include("royalty.jl")
include("production.jl")
include("drilling.jl")

# include("overall-likelihood.jl")

end # module
