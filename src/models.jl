abstract type AbstractModel end

import Base: length

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

# ----------------------------
# Overall structure
# ----------------------------

struct DataIndividual <:AbstractDataStructure end
struct DataSet <: AbstractDataStructure end
