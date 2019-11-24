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
using SparseArrays
using AxisAlgorithms

using LoopVectorization

# extend these methods
import Base: length, size, iterate,
    firstindex, lastindex, eachindex, getindex, IndexStyle,
    view, ==, eltype, +, -, isless,
    fill!,string, show, convert, eltype,
    step

# specific functions
using Distributions: _F1
using StatsBase: countmap, sample
using Base.Iterators: flatten, product, OneTo
using Dates: Month
using LinearAlgebra: checksquare, stride1

using InteractiveUtils: subtypes

# Real arrays
const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}

abstract type AbstractTmpVars end

# models
#---------------------

export AbstractModel, NoModel, AbstractDrillModel, AbstractProductionModel, AbstractRoyaltyModel, _nparm

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

peturb(x::T) where {T<:Real} = max( abs(x), one(T) ) * cbrt(eps(T))

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
include("drilling-model/abstract.jl")
include("drilling-model/test-reward-and-model.jl")

include("drilling-model/state-space.jl")

# reward function and components
include("drilling-model/drilling-reward.jl")
include("drilling-model/extend.jl")
include("drilling-model/cost.jl")
include("drilling-model/revenue.jl")

# dynamic model
include("drilling-model/dynamic-drilling-model.jl")
include("drilling-model/dcdp-components/makeIminusTVp.jl")
include("drilling-model/dcdp-components/learning_transition.jl")
include("drilling-model/dcdp-components/vfit.jl")
include("drilling-model/dcdp-components/solve-all-vfit.jl")

# ------------------------------------

# likelihoods
include("likelihood/royalty.jl")
include("likelihood/production.jl")
include("likelihood/drilling.jl")
include("likelihood/overall.jl")

end # module
