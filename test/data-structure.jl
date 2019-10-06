module ShaleDrillingLikelihood_DataStructure_Test

using ShaleDrillingLikelihood
using Test
using Random
using StatsBase

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, _num_choices, num_x


@testset "SimulationDraws" begin

    Random.seed!(1234)
    k = 3
    nobs = 10
    M = 5

    u = rand(M, nobs)
    v = rand(M, nobs)

    x = SimulationDraws(u,v)

    @test size(x) == (M,nobs)
    @test isa(x, SimulationDrawsMatrix)
    vw = view(x, 1)
    @test typeof(vw) <: SimulationDrawsVector
    @test size(vw) == (M,)
end


@testset "DataRoyalty" begin

    nobs = 10
    k = 3
    L = 3

    X = rand(k,nobs)
    y = collect(1:nobs)
    @views sample!(collect(1:L), y[L+1:end])

    data = DataRoyalty(y, X)

    @test length(data) == nobs
    @test size(data) == nobs
    @test num_x(data) == k
    @test_throws BoundsError data[nobs+1]
    @test_throws BoundsError data[0]
    @test iterate(data, nobs+1) == nothing
    @test iterate(data, 1) == iterate(data)
    let obsi = 0
        for obs in data
            obsi += 1
        end
        @test obsi == length(data)
    end
end



end # module
