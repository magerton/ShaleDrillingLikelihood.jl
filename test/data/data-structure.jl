module ShaleDrillingLikelihood_DataStructure_Test

using Revise
using ShaleDrillingLikelihood
using Test
using Random
using StatsBase
using InteractiveUtils
using StatsFuns

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    ObservationRoyalty, DataRoyalty, _y, _x, _xbeta, num_choices, _num_x,
    DataProduce, _xsum, obs_ptr, group_ptr, _nu,
    groupstart, grouplength, grouprange, obsstart, obsrange, obslength,
    ObservationProduce, ObservationGroup,
    _i, _data, _num_obs, update_nu!, Observation, update!,
    SimulationDraw,
    _ψ1, _ψ2, _ρ


@testset "SimulationDraws" begin

    Random.seed!(1234)
    k = 3
    nobs = 10
    M = 5

    u = rand(M, nobs)
    v = rand(M, nobs)
    qm = zeros(M)
    x = SimulationDraws(u,v, similar(u), similar(v), qm, similar(qm), similar(qm), similar(qm), reshape(qm,1,:))

    @test size(x) == (M,nobs)
    @test isa(x, SimulationDrawsMatrix)
    vw = view(x, 1)
    @test isa(vw, SimulationDrawsVector)
    @test size(vw) == (M,)

    @test size(SimulationDraws(M, nobs, 1)) == (M,nobs,)

    prodsim(s::SimulationDraw) = _ψ1(s) * _ψ2(s)

    N = 100_000
    thetarho = 0.5
    xbar = mean(prodsim(SimulationDraw(randn(), randn(), thetarho)) for i in 1:N)
    xtrue = _ρ(thetarho)
    @test abs(xbar-xtrue) < 0.01
end


@testset "DataRoyalty" begin

    nobs = 10
    k = 3
    rates = [1/8, 3/16, 1/4]
    L = length(rates)

    X = rand(k,nobs)
    y = collect(1:nobs)
    @views sample!(collect(1:L), y[L+1:end])

    data = DataRoyalty(RoyaltyModel(), y, X, rates)

    @test length(data) == nobs == _num_obs(data)
    @test _num_x(data) == k
    @test_throws BoundsError data[nobs+1]
    @test_throws BoundsError data[0]
    @test iterate(data, nobs+1) == nothing
    @test iterate(data, 1) == iterate(data)
    let obsi = 0
        for (i,grp) in enumerate(data)
            obsi += 1
            @test Observation(grp,1) == Observation(data,i) # == first(grp) == last(grp)
        end
        @test obsi == length(data)
    end
end

@testset "DataProduce" begin

    @testset "DataProduce Deterministic" begin
        n = 10
        k = 2
        nwell = 3
        ngroups = 3
        nobs = n*nwell
        beta = rand(2)
        alphapsi = 0.5
        sigeta = 0.3
        sigu = 0.5
        theta = vcat(alphapsi, beta, sigeta, sigu)

        x = rand(k, nobs)
        y = x' * beta + rand(nobs)

        xsum = hcat(collect(sum(x[:, (j-1)*n .+ (1:n)]; dims=2) for j in 1:nwell)...)
        xpnu = similar(xsum)
        nusum = zeros(size(xsum,2))
        nusumsq = similar(nusum)

        nu = y - x'*beta
        obsptr = [j*n+1 for j in 0:nwell]
        groupptr = vcat(1,2,fill(nwell+1,ngroups+1-2))

        d = DataProduce(ProductionModel(),y,x,xsum,nu,xpnu,nusum,nusumsq,obsptr,groupptr)
        update!(d, theta)
        obs = Observation(d,2)
        @test isa(obs, ObservationProduce)

        @test length(d) == ngroups == length(groupptr)-1
        @test _num_x(d) == k
        @test length(d) == last(groupptr)-1 == length(obsptr)-1

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

        grp = ObservationGroup(d,1)
        grouprange(grp)
        @test Observation(grp,1) == Observation(d,1)
        @test iterate(grp) == (Observation(d,1), 2,)
        @test iterate(grp, iterate(grp)[2]) == nothing

        grp = ObservationGroup(d,2)
        @test length(grp) == 2
        @test Observation(grp,1) == Observation(d,2)
        @test Observation(grp,2) == Observation(d,3)
        @test iterate(grp) == (Observation(d,2), 2,)
        @test iterate(grp, 2) == (Observation(d,3), 3,)
        @test iterate(grp, 3) == nothing

        grp = ObservationGroup(d,3)
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

    @testset "DataProduce random" begin
        numgroups = 11
        maxwells = 5
        mint = 4
        maxt = 25
        k = 3

        beta = rand(3)
        alphapsi = 0.5
        sigeta = 0.3
        sigu = 0.5

        # create group variables
        psi = rand(numgroups)
        grouplens = vcat(0, collect(0:maxwells)..., sample(0:maxwells, numgroups-maxwells-1))
        @test length(grouplens) == numgroups+1
        groups = 1 .+ cumsum(grouplens)

        @test all(groups[2:end] .- groups[1:end-1] .== grouplens[2:end])

        # create wells per group
        nwells = last(groups)-1
        us = rand(nwells)
        obslens = vcat(0, sample(mint:maxt, nwells))
        obsptr = 1 .+ cumsum(obslens)
        nobs = last(obsptr)-1

        @test all(obsptr[2:end] .- obsptr[1:end-1] .== obslens[2:end])

        # create observation epsilons
        e = rand(nobs)
        x = rand(k,nobs)
        xsum = zeros(k, nwells)
        xpnu = similar(xsum)
        nusum = zeros(size(xsum,2))
        nusumsq = similar(nusum)
        nu = zeros(nobs)
        y = x'*beta

        for i = 1:numgroups
            wellptrs = groups[i]:groups[i+1]-1
            for k in wellptrs
                obsrng = obsptr[k]:obsptr[k+1]-1
                # sum!(xsum[:,k:k], x[:, obsrng])
                # y[obsrng] .+= sigu .* us[k] .+ alphapsi .* psi[i]
            end
        end

        data = DataProduce(ProductionModel(),y,x,xsum,nu,xpnu,nusum,nusumsq,obsptr,groups)

        for g in data
            i = _i(g)
            for (k,o) in enumerate(g)
                j = getindex(grouprange(g), k)
                let y = _y(o)
                    y .+= sigu .* us[j] .+ alphapsi .* psi[i]
                end
                sum!(reshape(_xsum(o), :, 1), _x(o))
                _nu(o) .= _y(o) .- _x(o)'*beta
            end
        end

        for j = 1:nwells
            o = Observation(data,j)
            @test all(_xsum(o) .== sum(_x(o); dims=2))
        end

        @test _y(data) - _x(data)'*beta == _nu(data)

        @test !all(_nu(data) .== 0)

        dd = DataProduce(numgroups, maxwells, mint:maxt, vcat(rand(3), 0.5, 0.5, 0.3))

        # @code_warntype ShaleDrillingLikelihood.update_xsum!(dd)
        # @code_warntype ShaleDrillingLikelihood.update_xpnu!(dd)
        # @code_warntype update_nu!(dd, ProductionModel(), vcat(rand(3), 0.5, 0.5, 0.3))

    end
end


end # module
