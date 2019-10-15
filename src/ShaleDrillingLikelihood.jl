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

export AbstractModel, BigModel, royalty, drill, produce

"Holds triple of statistical models"
struct BigModel{A<:AbstractModel,B<:AbstractModel,C<:AbstractModel} <: AbstractModel
    royalty::A
    drill::B
    produce::C
end
royalty(m::BigModel) = m.royalty
drill(  m::BigModel) = m.drill
produce(m::BigModel) = m.produce

"No model"
struct NoModel <: AbstractModel end

# Overall structure
#----------------------------

# data structures
include("simulation-data-structure.jl")
include("abstract-data-structure.jl")

include("royalty-data-structure.jl")
include("production-data-structure.jl")
include("drilling-data-structure.jl")

include("sum-functions.jl")

include("flow.jl")

# likelihoods
include("royalty.jl")
include("production.jl")
include("drilling.jl")

include("overall-likelihood.jl")

end # module
