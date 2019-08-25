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
num_choices(m::Union{RoyaltyModelNoHet,RoyaltyModel}) = m.num_choices
num_x(m::Union{RoyaltyModelNoHet,RoyaltyModel}) = m.num_x

"checks if the choice is valid"
function choice_in_model(m::AbstractRoyaltyModel, l::Integer)
    0 < l <= num_choices(m) && return true
    throw(DomainError(l, "$l outside of 1:$(num_choices(m))"))
end

# length of RoyaltyModel parameters & gradient
length(m::RoyaltyModelNoHet) = num_x(m) + num_choices(m) - 1
length(m::RoyaltyModel)      = num_x(m) + num_choices(m) + 1

# parameter vector
idx_roy_ρ(m::RoyaltyModelNoHet) = 1:0
idx_roy_ψ(m::RoyaltyModelNoHet) = 1:0
idx_roy_β(m::RoyaltyModelNoHet) = (1:num_x(m))
idx_roy_κ(m::RoyaltyModelNoHet) = num_x(m) .+ (1:num_choices(m)-1)
function idx_roy_κ(m::RoyaltyModelNoHet, l::Integer)
    choice_in_model(m,l)
    return num_x(m) + l
end

idx_roy_ρ(m::RoyaltyModel) = 1
idx_roy_ψ(m::RoyaltyModel) = 2
idx_roy_β(m::RoyaltyModel) = 2 .+ (1:num_x(m))
idx_roy_κ(m::RoyaltyModel) = 2 + num_x(m) .+ (1:num_choices(m)-1)
function idx_roy_κ(m::RoyaltyModel, l::Integer)
    choice_in_model(m,l)
    2 + num_x(m) + l
end

# get coefs
theta_roy(  m::AbstractRoyaltyModel, theta) = theta
theta_roy_ρ(m::AbstractRoyaltyModel, theta) = theta[idx_roy_ρ(m)]
theta_roy_ψ(m::AbstractRoyaltyModel, theta) = theta[idx_roy_ψ(m)]
theta_roy_β(m::AbstractRoyaltyModel, theta) = view(theta, idx_roy_β(m))
theta_roy_κ(m::AbstractRoyaltyModel, theta) = view(theta, idx_roy_κ(m))
theta_roy_κ(m::AbstractRoyaltyModel, theta, l) = theta[idx_roy_κ(m,l)]

dpsidrhom(m::AbstractRoyaltyModel, theta, ψ0)  = 0.0


# check if theta is okay
theta_roy_check(m::AbstractRoyaltyModel, theta) = issorted(theta_roy_κ(m,theta))
