abstract type AbstractModel end

import Base: length

export AbstractModel, BigModel,
    AbstractRoyaltyModel, NoHet, RoyaltyModel,
    ProductionModel,
    royalty, drill, produce


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

# ----------------------------------------------
# Royalty Model
# ----------------------------------------------

"Royalty rates"
abstract type AbstractRoyaltyModel <: AbstractModel end
struct RoyaltyModelNoHet           <: AbstractRoyaltyModel end
struct RoyaltyModel                <: AbstractRoyaltyModel end



# ----------------------------------------------
# Production Model
# ----------------------------------------------

"Production"
struct ProductionModel <: AbstractModel
    num_x::Int
end

num_x(m::ProductionModel) = m.num_x
length(m::ProductionModel) = num_x(m) + 3

idx_produce_ψ(  m::ProductionModel) = 1
idx_produce_β(  m::ProductionModel) = 1 .+ (1:num_x(m))
idx_produce_σ2η(m::ProductionModel) = 2 + num_x(m)
idx_produce_σ2u(m::ProductionModel) = 3 + num_x(m)

theta_produce(    m::ProductionModel, theta) = theta
theta_produce_ψ( m::ProductionModel, theta) = theta[idx_produce_ψ( m)]
theta_produce_β(  m::ProductionModel, theta) = view(theta, idx_produce_β(m))
theta_produce_σ2η(m::ProductionModel, theta) = theta[idx_produce_σ2η(m)]
theta_produce_σ2u(m::ProductionModel, theta) = theta[idx_produce_σ2u(m)]
