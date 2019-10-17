
# Simulation of data
# -----------------------------------------------------

function payoff(d::Integer, psi::Real,x::Real,z::Tuple{Real},theta::AbstractVector)
    0 <= d <= 2 || throw(DomainError())
    length(theta)==3 || throw(DimensionMismatch())
    out = d*(theta[1]*psi + theta[2]*x + theta[3]*first(z))
    return Float64(out)
end

is_development(lease::ObservationGroup{<:DrillInitial}) = false
is_development(lease::ObservationGroup{<:DrillDevelopment}) = true

function simulate_lease(lease::DrillLease, theta::AbstractVector{<:Number})
    nper = length(lease)
    zc = zchars(lease)
    ic = ichars(lease)
    x = _x(lease)
    y = _y(lease)
    x .= (1 + is_development(lease))

    i = uniti(lease)
    m = _model(DataDrill(lease))
    sim = SimulationDraw(theta_drill_ρ(m,theta))

    ubv = Vector{Float64}(undef, length(actionspace(m))

    x[1] = initial_state(m)
    for t in 1:nper
        obs = ObservationDrill(m, ic, zc[t], y[t], x[t])
        f(d) = flow(d,obs,theta,sim)
        payoffs .= f.(actionspace(obs))
        softmax!(payoffs)
        cumsum!(payoffs, payoffs)
        choice = searchsortedfirst(payoffs, rand())-1
        y[t] = choice
        t < nper && x[t+1] = next_state(m,choice)
    end
end
