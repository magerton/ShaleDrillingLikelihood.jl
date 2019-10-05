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

struct RoyaltyModelNoHet <: AbstractRoyaltyModel
    num_choices::Int
    num_x::Int
end

struct RoyaltyModel <: AbstractRoyaltyModel
    num_choices::Int
    num_x::Int
end

# access fields
num_choices(m::AbstractRoyaltyModel) = m.num_choices
num_x(m::AbstractRoyaltyModel) = m.num_x

"checks if the choice is valid"
function choice_in_model(m::AbstractRoyaltyModel, l::Integer)
    0 < l <= num_choices(m) && return true
    throw(DomainError(l, "$l outside of 1:$(num_choices(m))"))
end

# length of RoyaltyModel parameters & gradient
length(m::RoyaltyModelNoHet) = num_x(m) + num_choices(m) - 1
length(m::RoyaltyModel)      = num_x(m) + num_choices(m) + 1

# parameter vector
idx_royalty_ρ(m::RoyaltyModelNoHet) = 1:0
idx_royalty_ψ(m::RoyaltyModelNoHet) = 1:0
idx_royalty_β(m::RoyaltyModelNoHet) = (1:num_x(m))
idx_royalty_κ(m::RoyaltyModelNoHet) = num_x(m) .+ (1:num_choices(m)-1)
function idx_royalty_κ(m::RoyaltyModelNoHet, l::Integer)
    choice_in_model(m,l)
    return num_x(m) + l
end

idx_royalty_ρ(m::RoyaltyModel) = 1
idx_royalty_ψ(m::RoyaltyModel) = 2
idx_royalty_β(m::RoyaltyModel) = 2 .+ (1:num_x(m))
idx_royalty_κ(m::RoyaltyModel) = 2 + num_x(m) .+ (1:num_choices(m)-1)
function idx_royalty_κ(m::RoyaltyModel, l::Integer)
    choice_in_model(m,l)
    2 + num_x(m) + l
end

# get coefs
theta_roy(  m::AbstractRoyaltyModel, theta) = theta
theta_royalty_ρ(m::AbstractRoyaltyModel, theta) = theta[idx_royalty_ρ(m)]
theta_royalty_ψ(m::AbstractRoyaltyModel, theta) = theta[idx_royalty_ψ(m)]
theta_royalty_β(m::AbstractRoyaltyModel, theta) = view(theta, idx_royalty_β(m))
theta_royalty_κ(m::AbstractRoyaltyModel, theta) = view(theta, idx_royalty_κ(m))
theta_royalty_κ(m::AbstractRoyaltyModel, theta, l) = theta[idx_royalty_κ(m,l)]

dpsidrhom(m::AbstractRoyaltyModel, theta, ψ0)  = 0.0

# check if theta is okay
theta_royalty_check(m::AbstractRoyaltyModel, theta) = issorted(theta_royalty_κ(m,theta))


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
