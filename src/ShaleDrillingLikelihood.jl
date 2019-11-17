module ShaleDrillingLikelihood

# general libraries
using StatsFuns
using LinearAlgebra
using Base.Threads
using Halton
using Dates
using Calculus
using Distributions

using LoopVectorization

# extend these methods
import Base: length, size, iterate,
    firstindex, lastindex, eachindex, getindex, IndexStyle,
    view, ==, eltype, +, -, isless

# specific functions
using Distributions: _F1
using StatsBase: countmap, sample
using Base: OneTo
using Base.Iterators: flatten
using Dates: Month
using LinearAlgebra: checksquare


# Real arrays
const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

abstract type AbstractTmpVars end

# models
#---------------------

export AbstractModel, NoModel, AbstractDrillModel, AbstractProductionModel, AbstractRoyaltyModel

# for modeling
abstract type AbstractModel end

"No model"
struct NoModel <: AbstractModel end
length(::NoModel) = 0

abstract type AbstractDrillModel      <: AbstractModel end
abstract type AbstractProductionModel <: AbstractModel end
abstract type AbstractRoyaltyModel    <: AbstractModel end


# Overall structure
#----------------------------

include("sum-functions.jl")

# drilling model
include("drilling-model/models.jl")

# data structures
include("data/abstract.jl")
include("data/simulation.jl")
include("data/royalty.jl")
include("data/production.jl")
include("data/time-variable-type.jl")
include("likelihood/drilling-tmpvars.jl")
include("data/drilling.jl")
include("data/overall.jl")

# flow for drilling
include("drilling-model/flow.jl")

# likelihoods
include("likelihood/royalty.jl")
include("likelihood/production.jl")
include("likelihood/drilling.jl")
include("likelihood/overall.jl")

end # module
