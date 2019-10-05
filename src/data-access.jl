
# royalty.jl
_xroy(data, i) = view(data, :, i)


# drilling.jl
irng(num_t::Int, i::Int) = (i-1)*num_t .+ (1:num_t)


# unobservables we integrate out
_ψ1(u::Real, v::Real, ρ::Real) = ρ*u + sqrt(1-ρ^2)*v
_ψ2(u::Real, v::Real, ρ::Real) = _ψ2(u,v)
_ψ2(u::Real, v::Real) = u

_dψ1dρ(u::Real,v::Real,ρ::Real) = u - ρ/sqrt(1-ρ^2)*v

dpsidrho(model, theta, u, v) = _dψ1dρ(u,v,theta_royalty_ρ(model,theta))
