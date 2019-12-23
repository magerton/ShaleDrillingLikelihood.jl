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
using Interpolations
using LoopVectorization
using ProgressMeter
using GenGlobal
using SharedArrays
using Distributed
using CountPlus
using Optim
using StatsBase
using ClusterManagers
using DataFrames

using UnsafeArrays
using ArgParse

# extend these methods
import Base: length, size, iterate,
    firstindex, lastindex, eachindex, getindex, IndexStyle,
    view, ==, eltype, +, -, isless,
    fill!,string, show, convert, eltype,
    step

import StatsBase: coeftable, coefnames
import ArgParse: parse_item

export coefnames, print_in_binary_for_copy_paste

# specific functions
using Printf: @sprintf
using Distributions: _F1
using StatsBase: countmap, sample
using Base.Iterators: flatten, product, OneTo
using Dates: Month
using LinearAlgebra: checksquare, stride1
using Optim: minimizer

using InteractiveUtils: subtypes

# Real arrays
const AbstractRealArray{T,N} = AbstractArray{T,N} where {T<:Real,N}
const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}


abstract type AbstractTmpVars end

# models
#---------------------

export AbstractModel,
    NoModel,
    AbstractDrillModel, AbstractDynamicDrillModel, AbstractStaticDrillModel,
    AbstractProductionModel,
    AbstractRoyaltyModel,
    _nparm,
    _model,
    sprintf_binary,
    AbstractPayoffFunction,
    AbstractStaticPayoff,
    AbstractPayoffComponent,
    AbstractDrillingRevenue,
    AbstractDrillingCost,
    AbstractExtensionCost

# for modeling
abstract type AbstractModel end

"No model"
struct NoModel <: AbstractModel end
length(::NoModel) = 0
_nparm(::NoModel) = 0

abstract type AbstractProductionModel    <: AbstractModel end
abstract type AbstractRoyaltyModel       <: AbstractModel end
abstract type AbstractDrillModel         <: AbstractModel end
abstract type AbstractDynamicDrillModel  <: AbstractDrillModel end
abstract type AbstractStaticDrillModel   <: AbstractDrillModel end

abstract type AbstractModelVariations end

# Static Payoff
abstract type AbstractPayoffFunction end
abstract type AbstractStaticPayoff   <: AbstractPayoffFunction end
abstract type AbstractPayoffComponent <: AbstractPayoffFunction end

# payoff components
abstract type AbstractDrillingRevenue <: AbstractPayoffComponent end
abstract type AbstractDrillingCost    <: AbstractPayoffComponent end
abstract type AbstractExtensionCost   <: AbstractPayoffComponent end

# also needed
abstract type AbstractStateSpace end
abstract type AbstractUnitProblem <: AbstractStateSpace end


# data structures
#---------------------

# For data structures

export AbstractDataDrill

# "Collection of data on a particular outcome for individuals `i`"
# "What we feed into a likelihood"
# "Group of observations"
abstract type AbstractDataStructure end
abstract type AbstractDataSetofSets    <: AbstractDataStructure end

abstract type AbstractDataSet          <: AbstractDataStructure end
abstract type AbstractObservationGroup <: AbstractDataStructure end
abstract type AbstractObservation      <: AbstractDataStructure end

abstract type AbstractDataDrill <: AbstractDataSet end


const DataOrObs = Union{AbstractDataSet,AbstractObservation}

_model(x::AbstractDataStructure) = x.model

export EmptyDataSet

"Empty Data Set"
struct EmptyDataSet <: AbstractDataSet end
length(d::EmptyDataSet) = 0
eachindex(d::EmptyDataSet) = 1:typemax(Int)
_nparm(d::EmptyDataSet) = 0
_model(d::EmptyDataSet) = NoModel()
coefnames(d::EmptyDataSet) = Vector{String}(undef,0)

# price / cost  / year
#----------------------------

# access Zchars
const PriceTuple         = Tuple{Float64}
const PriceYearTuple     = Tuple{Float64, <:Integer}
const PriceCostYearTuple = Tuple{Float64, Float64, <:Integer}
const PriceCostTuple     = Tuple{Float64, Float64}

# zchars
second(z::Tuple) = getindex(z,2)
logprice(  z::NTuple{N,Real}) where {N} = first(z)
logrigrate(z::NTuple{N,Real}) where {N} = second(z)
year(      z::NTuple{N,Real}) where {N} = last(z)

# useful functions
#----------------------------

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

function sprintf_binary(x::Vector{<:Number})
    xstr = reduce(*, @sprintf("%a, ", i) for i in x)
    return "[" * xstr * "]"
end
@deprecate print_in_binary_for_copy_paste(x) sprintf_binary(x)

check_finite(x::AbstractArray) = all(isfinite.(x)) || throw(error("x not finite!"))
check_finite(x::AbstractVector) = all(isfinite.(x)) || throw(error("x not finite! $x"))
check_finite(x::Number) = isfinite(x) || throw(error("x not finite! $x"))


all_same_value(x) = all(x .== first(x))
range_i_to_ip1(x,i) = x[i] : (x[i+1]-1)


# Overall structure
#----------------------------

include("utilities/threadutils.jl")
include("utilities/sum-functions.jl")
include("utilities/inplace-interpolation.jl")
include("utilities/delegate-methods.jl")
include("utilities/helper-macros.jl")

include("time-variables/tvpack.jl")
include("time-variables/tauchen.jl")
include("time-variables/time-variable-type.jl")
include("time-variables/time-series-processes.jl")

# data structures
include("data/observation-group.jl")
include("data/simulation-draws.jl")
include("data/royalty.jl")
include("data/production.jl")

include("data/drilling-data-tmpvars.jl")
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
include("drilling-model/dynamic-drilling-tmpvars.jl")
include("drilling-model/value-function.jl")
include("drilling-model/dynamic-drilling-model.jl")

# VF iteration components
include("drilling-model/dcdp-components/makeIminusTVp.jl")
include("drilling-model/dcdp-components/learning_transition.jl")
include("drilling-model/dcdp-components/vfit.jl")
include("drilling-model/dcdp-components/solve-all-vfit.jl")

# simulation
include("drilling-model/data-simulation.jl")

# Likelihood / solution
# ------------------------------------

# likelihoods
include("likelihood/royalty.jl")
include("likelihood/production.jl")
include("likelihood/drilling.jl")
include("likelihood/overall.jl")

# solution
include("likelihood/nfxp.jl")
include("likelihood/optimize.jl")
include("likelihood/display-results.jl")

# parameters, model generation
include("SDLParameters/SDLParameters.jl")

# counterfactuals
include("counterfactuals/shared-simulations.jl")
include("counterfactuals/objects.jl")
include("counterfactuals/sparse-state-transition.jl")
include("counterfactuals/simulate-lease.jl")
include("counterfactuals/simulate-unit.jl")
include("counterfactuals/wrapper.jl")

end # module
