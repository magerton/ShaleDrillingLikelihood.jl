module ShaleDrillingLikelihood_Drilling_Test

# using Revise
using ShaleDrillingLikelihood

using Test
using StatsFuns
using Random
using BenchmarkTools
using Base.Threads
# using Profile
# using ProfileView
using BenchmarkTools
# using InteractiveUtils

using Calculus
using Optim
using LinearAlgebra

using ShaleDrillingLikelihood: SimulationDraws,
    DataDrill,
    TestDrillModel,
    DrillingTmpVars,
    simloglik_drill!,
    DrillUnit,
    AbstractDrillRegime,
    DrillLease,
    ObservationDrill,
    InitialDrilling,
    DevelopmentDrilling,
    theta_drill_ρ,
    _model,
    update!,
    logL,
    _y


println("testing drilling likelihood")

@testset "Drilling Likelihood" begin

    Random.seed!(1234)

    theta = [1.0, 1.0, 1.0, 0.5]

    data = DataDrill(
        TestDrillModel(), theta;
        minmaxleases=10:100,
        num_i=1_000, nperinitial=1:30, nper_development=0:40,
        num_zt=200
    )

    sim = SimulationDraws(500, data)
    println("number of periods is $(length(_y(data)))")

    grad = zeros(length(theta))
    dtv = DrillingTmpVars(data, theta)

    LL1 = logL(data,sim,dtv,theta)
    LL2 = logL(data,sim,dtv,theta)
    @test isfinite(LL1)
    @test isfinite(LL2)
    @test LL1 ≈ LL2

    @show @benchmark logL($data,$sim,$dtv,$theta)

    # # -----------------------------------------------
    # # simulate data
    # # -----------------------------------------------
    #
    #
    # num_i = 100
    # num_t = 50
    # nobs = num_i *num_t
    # β = [1.0, -2.0, 1.0, 2.0]
    # k = length(β)
    # L = 3
    #
    # # exogenous variable
    # X = randn(nobs)
    # psi = repeat(randn(num_i), inner=num_t)
    #
    # # choice utilities
    # u = vcat(zeros(nobs)', β[1:2]*X' .+ β[3:4]*psi')
    #
    # # multinomial logit probabilities
    # Pr0 = mapslices(softmax, u; dims=1)
    # cum_Pr0 = cumsum(Pr0; dims=1)
    #
    # # random choice given multinomial probabilites
    # e_quantile = rand(nobs)
    # choices = [searchsortedfirst(view(cum_Pr0, :, i), e_quantile[i]) for i in 1:length(e_quantile) ]
    #
    # # -----------------------------------------------
    # # form likelihood
    # # -----------------------------------------------
    #
    # theta = rand(k)
    # ubV = zeros(L, nobs)
    # drng = 1:L
    # M = 10*nthreads()
    # psisim = randn(M,num_i)
    #
    # grad = similar(theta)
    #
    # println("run threaded once")
    # llthrd = simloglik_drill!(grad, choices, X, psisim, theta, num_t, num_i)
    # gradan = copy(grad)
    # gradfd = Calculus.gradient( θ -> simloglik_drill!(Vector{Float64}(undef,0), choices, X, psisim, θ, num_t, num_i), theta )
    # @test gradan ≈ gradfd
    #
    # # @code_warntype simloglik_drill!(grad, choices, X, psisim, theta, num_t, num_i)
    #
    # println("\n\n-------- Threaded using $(nthreads()) threads --------\n\n")
    # # @show @btime simloglik_drill!($grad, $choices, $X, $psisim, $theta, $num_t, $num_i)

end # testset
end # module
