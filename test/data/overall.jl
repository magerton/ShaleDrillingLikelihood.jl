module ShaleDrillingLikelihood_OverallLikelihood_Test

# using Revise
# using Juno
# using Profile
# using ProfileView

using ShaleDrillingLikelihood
using Test
using StatsFuns
using Calculus
using Optim
using Random
using InteractiveUtils
using LinearAlgebra
using BenchmarkTools

using Base: OneTo

using ShaleDrillingLikelihood: AbstractDataSet,
    DataRoyalty, DataProduce, DataDrill,
    SimulationDraws,
    _nparm,
    TestDrillModel,
    theta_drill_ρ,
    theta_drill_ψ,
    theta_royalty_ρ,
    theta_produce_ψ,
    idx_drill, theta_drill,
    idx_royalty, theta_royalty,
    idx_produce, theta_produce,
    idx_drill_ρ, idx_royalty_ρ,
    idx_drill_ψ, idx_produce_ψ,
    thetas

println("testing overall royalty")

@testset "Access parameter vectors of joint model(s)" begin

    num_i = 500
    L = 3
    M = 500
    u = randn(num_i)
    v = randn(num_i)
    psi2 = u

    sim = SimulationDraws(M, num_i)

    # set up coefs
    θρ = 0.5
    αψ = 0.6

    # set up drilling coefs
    model_drill = TestDrillModel()
    θ_drill   = [αψ, 2.0, -2.0, -0.75, θρ]
    @test theta_drill_ρ(model_drill, θ_drill) == θρ
    @test theta_drill_ψ(model_drill, θ_drill) == αψ

    # set up royalty coefs
    θ_royalty = [θρ, 1.0,    -1.0, 1.0, 1.0,    -0.6, 0.6]  # dψdρ, ψ, β, κ
    @test theta_royalty_ρ(RoyaltyModel(), θ_royalty) == θρ

    # set up pdxn coefs
    θ_produce = vcat(αψ, rand(3), 0.3, 0.4)
    @test theta_produce_ψ(ProductionModel(), θ_produce) == αψ

    # big theta
    θ = vcat(θ_drill, θ_royalty[2:end], θ_produce[2:end])
    @test theta_drill(model_drill, θ) == θ_drill

    # make data
    data_drill_opt = (num_zt=200, minmaxleases=1:2, nper_initial=10:20, nper_development=0:10, tstart=1:50,)
    data_drill   = DataDrill(u, v, TestDrillModel(), θ_drill; data_drill_opt...)
    data_produce = DataProduce(u, 10, 10:20, θ_produce)
    data_royalty = DataRoyalty(u,v,θ_royalty,L)

    data = DataSetofSets(data_drill, data_royalty, data_produce)
    dataroypdxn = DataSetofSets(EmptyDataSet(), data_royalty, data_produce)
    datadrillonly = DataSetofSets(data_drill, EmptyDataSet(), EmptyDataSet() )

    @test length(data_drill) == num_i
    @test length(data_royalty) == num_i
    @test length(data_produce) == num_i
    @test all(length.(data) .== num_i)
    @test all(_nparm.(data) .== (_nparm(data_drill), _nparm(data_royalty), _nparm(data_produce)))

    @test idx_produce_ψ isa Function
    @test idx_drill_ψ isa Function
    coef_links = [(idx_produce_ψ, idx_drill_ψ,),]
    @test coef_links isa Vector{<:NTuple{2,Function}}

    @test (θ_drill, θ_royalty, θ_produce) == thetas(data, vcat(θ_drill, θ_royalty[2:end], θ_produce[2:end]), coef_links)
    @test (θ_drill, θ_royalty, θ_produce) == thetas(data, vcat(θ_drill, θ_royalty[2:end], θ_produce))
    @test (θ_drill, θ_royalty, θ_produce) == thetas(data, vcat(θ_drill, θ_royalty[2:end], θ_produce), ())
    @test (θ_drill, θ_royalty, θ_produce) == thetas(data, vcat(θ_drill, θ_royalty[2:end], θ_produce), [])
    @test ([], θ_royalty, θ_produce,) == thetas(dataroypdxn, vcat(θ_royalty, θ_produce))
    @test (θ_drill, [], []) == thetas(datadrillonly, θ_drill)


    @test theta_drill(  data, θ) == θ_drill
    @test theta_royalty(data, θ) == θ_royalty
    @test theta_produce(data, θ, coef_links) == θ_produce
end

end # module
