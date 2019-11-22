module ShaleDrillingLikelihood

# general libraries
using StatsFuns
using LinearAlgebra
using Base.Threads
using Halton
using Dates
using Calculus
using Distributions
using Test

using LoopVectorization

# extend these methods
import Base: length, size, iterate,
    firstindex, lastindex, eachindex, getindex, IndexStyle,
    view, ==, eltype, +, -, isless,
    fill!,string, show, convert

# specific functions
using Distributions: _F1
using StatsBase: countmap, sample
using Base: OneTo
using Base.Iterators: flatten
using Dates: Month
using LinearAlgebra: checksquare

using InteractiveUtils: subtypes

# Real arrays
const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}
const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}


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

abstract type AbstractModelVariations end


# to get a scalar one or zero
aone(x) = one(eltype(x))
azero(x) = zero(eltype(x))



function showtypetree(T, level=0)
    println("\t" ^ level, T)
    for t in subtypes(T)
        showtypetree(t, level+1)
   end
end


# Overall structure
#----------------------------

include("threadutils.jl")
include("sum-functions.jl")


# data structures
include("data/abstract.jl")
include("data/simulation.jl")
include("data/royalty.jl")
include("data/production.jl")
include("data/time-variable-type.jl")
include("likelihood/drilling-tmpvars.jl")
include("data/drilling.jl")
include("data/overall.jl")

# drilling model
# ------------------------------------
include("drilling-model/abstract-drilling-model.jl")
include("drilling-model/test-drilling-model.jl")
include("drilling-model/drilling-model.jl")

include("drilling-model/state-space.jl")

include("drilling-model/reward-functions.jl")
include("drilling-model/extend.jl")
include("drilling-model/cost.jl")
include("drilling-model/revenue.jl")

include("drilling-model/tempvars.jl")

# ------------------------------------

# likelihoods
include("likelihood/royalty.jl")
include("likelihood/production.jl")
include("likelihood/drilling.jl")
include("likelihood/overall.jl")

end # module
