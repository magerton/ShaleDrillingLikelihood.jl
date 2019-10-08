module ShaleDrillingLikelihood

using StatsFuns
using Distributions: _F1
using LinearAlgebra
using Base.Threads
using Halton

import Base: length, size, iterate, firstindex, lastindex, getindex, IndexStyle, view, ==
using StatsBase: countmap, sample

# for computations
abstract type AbstractIntermediateComputations end
abstract type AbstractTempVar end

# For data structures
abstract type AbstractDataStructure end
abstract type AbstractObservationGroup end
abstract type AbstractObservation end

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

struct DataIndividual <:AbstractDataStructure end
struct DataSet <: AbstractDataStructure end


# data structures
include("simulation-data-structure.jl")

include("royalty-data-structure.jl")
include("production-data-structure.jl")
include("drilling-data-structure.jl")

include("sum-functions.jl")

include("flow.jl")

# likelihoods
include("royalty.jl")
include("production.jl")
include("drilling.jl")

# include("overall-likelihood.jl")

end # module
