module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra
using Base.Threads
using Halton
using Dates

import Base: length, size, iterate,
    firstindex, lastindex, eachindex, getindex, IndexStyle,
    view, ==, eltype, +, -, isless
using StatsBase: countmap, sample
using Base: OneTo
using Base.Iterators: flatten
using Dates: Month

using LinearAlgebra: checksquare

# for computations
abstract type AbstractIntermediateComputations end
abstract type AbstractTempVar end

# for modeling
abstract type AbstractModel end

# Real arrays
const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

# models
#---------------------

"No model"
struct NoModel <: AbstractModel end

# Overall structure
#----------------------------

include("sum-functions.jl")
include("flow.jl")

# data structures
include("data/simulation.jl")
include("data/abstract.jl")
include("data/royalty.jl")
include("data/production.jl")
include("data/drilling.jl")

# likelihoods
include("likelihood/royalty.jl")
include("likelihood/production.jl")
include("likelihood/drilling.jl")
include("likelihood/overall.jl")

end # module
