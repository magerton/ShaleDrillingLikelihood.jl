module ShaleDrillingLikelihood_DataStructure_Test

using Revise
using ShaleDrillingLikelihood
using Test
using Random
using StatsBase

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, _num_choices, _num_x,
    DataProduce, _xsum, obs_ptr, group_ptr, _nu,
    groupstart, grouplength, grouprange, obsstart, obsrange, obslength,
    ObservationProduce, ObservationGroupProduce,
    _i, _data, _num_obs


@testset "SimulationDraws" begin

    Random.seed!(1234)
    k = 3
    nobs = 10
    M = 5

    u = rand(M, nobs)
    v = rand(M, nobs)

    x = SimulationDraws(u,v, similar(u), similar(v))

    @test size(x) == (M,nobs)
    @test isa(x, SimulationDrawsMatrix)
    vw = view(x, 1)
    @test isa(vw, SimulationDrawsVector)
    @test size(vw) == (M,)

    @test size(SimulationDraws(M, nobs)) == (M,nobs,)

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
    @test _num_x(data) == k
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

@testset "DataProduce" begin

    n = 10
    k = 2
    nwell = 3
    ngroups = 3
    nobs = n*nwell
    beta = rand(2)

    x = rand(k, nobs)
    y = x' * beta + rand(nobs)

    xsum = hcat(collect(sum(x[:, (j-1)*n .+ (1:n)]; dims=2) for j in 1:nwell)...)

    nu = y - x'*beta
    obsptr = [j*n+1 for j in 0:nwell]
    groupptr = vcat(1,2,fill(nwell+1,ngroups+1-2))

    d = DataProduce(y,x,xsum,nu,obsptr,groupptr)
    obs = ObservationProduce(d,2)

    @test length(d) == ngroups == length(groupptr)-1
    @test _num_x(d) == k

    @test obsstart(d,2) == n+1
    @test obsstart(d,3) == 2*n + 1
    rng = obsstart(d,2) : obsstart(d,3)-1
    @test isa(rng, UnitRange)
    @test obsrange(d,2) == rng

    @test obslength(d,2) == length(rng)
    @test obsstart(d,2) == first(rng)

    @test length(obs) == n
    @test _y(obs) == y[rng]
    @test _x(obs) == x[:, rng]
    @test dropdims(sum(_x(obs); dims=2); dims=2) == _xsum(obs)

    grp = ObservationGroupProduce(d,1)
    grouprange(grp)
    @test ObservationProduce(grp,1) == ObservationProduce(d,1)
    @test iterate(grp) == (ObservationProduce(d,1), 2,)
    @test iterate(grp, iterate(grp)[2]) == nothing

    grp = ObservationGroupProduce(d,2)
    @test length(grp) == 2
    @test ObservationProduce(grp,1) == ObservationProduce(d,2)
    @test ObservationProduce(grp,2) == ObservationProduce(d,3)
    @test iterate(grp) == (ObservationProduce(d,2), 2,)
    @test iterate(grp, 2) == (ObservationProduce(d,3), 3,)
    @test iterate(grp, 3) == nothing

    grp = ObservationGroupProduce(d,3)
    @test length(grp) == 0
    @test iterate(grp) == nothing

    ii, jj = (0,0)
    for g in d
        ii += 1
        for o in g
            jj+=1
        end
    end
    @test ii == length(d) == ngroups == length(groupptr)-1
    @test jj == _num_obs(d) == nwell

end


end # module
