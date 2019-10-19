module ShaleDrillingLikelihood_DrillingDataStructure_Test

using ShaleDrillingLikelihood
using Test
using Random
using StatsBase
using StatsFuns
using InteractiveUtils
using Dates
using BenchmarkTools

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    AbstractDrillModel, DrillModel, TestDrillModel,
    ExogTimeVars, _timestamp, _timevars, Quarter,
    j1ptr, j2ptr, tptr, zchars, jtstart, ichars, _y, _x, j1chars, hasj1ptr, maxj1length,
    j1start, j1stop, j1length, j1_range,
    tstart, tstop, trange, tlength,
    zcharsvec,
    AbstractDataDrill, DataDrill,
    ObservationGroup,
    Observation, ObservationDrill,
    DrillUnit,
    AbstractRegimeType, InitialDrilling, DevelopmentDrilling, FinishedDrilling,
    AbstractDrillRegime, DrillInitial, DrillDevelopment, DrillLease,
    _i, _data,
    ObservationGroup,
    action, state,
    uniti,
    simulate_lease,
    randsumtoone,
    num_initial_leases



@testset "Drilling Data Structure" begin

    @testset "Exog Time Vars" begin
        nt = 12
        startdate = Date(2003,10)
        daterange = range(startdate; step=Quarter(1), length=nt)
        etv = ExogTimeVars([(x,) for x in randn(nt)], daterange)
        @test (12,1,) == size(etv)
        @test etv[2] == etv[Date(2004,1)]
        @test_throws DomainError etv[Date(2004,2)]
    end

    @testset "initial vs dev't" begin
        @test InitialDrilling()+1 == InitialDrilling()+100 == DevelopmentDrilling()
        @test InitialDrilling()-1 == nothing
        @test DevelopmentDrilling()+1 == FinishedDrilling()
        @test DevelopmentDrilling()-1 == InitialDrilling()
        @test FinishedDrilling()+1 == nothing
        @test FinishedDrilling()-1 == DevelopmentDrilling()
        @test InitialDrilling() in (InitialDrilling(), DevelopmentDrilling())
    end

    @testset "DataDrill Construction" begin
        nt = 12
        _j1ptr   = [1, 1]
        _j2ptr   = 1:1
        _tptr    = [1, 5]
        _jtstart = [1,]
        _jchars  = zeros(Int,0)
        _ichars  = [(1.0,), ]
        y = collect(1:4)
        x = collect(1:4)
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))
        _zchars_short  = ExogTimeVars([(x,) for x in randn(2)], range(Date(2003,10); step=Quarter(1), length=2))

        DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)
        @test_throws DimensionMismatch DataDrill(DrillModel(), [1,], _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)
        @test_throws ErrorException    DataDrill(DrillModel(),       _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars_short)
    end


    @testset "Iteration over DataDrill" begin
        nt = 12
        _j1ptr   = [1, 2]
        _j2ptr   = 2:2
        _tptr    = [1, 5, 6]
        _jtstart = [1, 5]
        _jchars  = [1.0, ]
        _ichars  = [(1.0,), ]
        _tchars  = [(x,x,) for x in 1:5]
        y = collect(1:5)
        x = copy(y)
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))

        data = DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)

        @test length(data) == 1
        @test maxj1length(data) == 1

        @test j1ptr(data, 1) == 1
        @test j1_range(data,1) == 1:1
        @test j1length(data,1) == 1
        @test j1stop(data,1) == 1
        @test hasj1ptr(data) == true

        @test tstart(data,1) == 1
        @test tstop(data,1) == 4
        @test trange(data,1) == 1:4
        @test tlength(data,1) == 4

        @test tstart(data,2) == 5
        @test tstop(data,2) == 5
        @test trange(data,2) == 5:5
        @test tlength(data,2) == 1

        @test zcharsvec(data, 2) == _timevars(_zchars)[2:end]

        # iterate through all
        i,r,l,t = (0,0,0,0)

        for unit in data
            i+= 1
            @test _i(unit) == i
            @test isa(unit, DrillUnit)
            @test DataDrill(unit) === data
            @test length(unit) == DevelopmentDrilling()
            @test num_initial_leases(unit) == j1length(unit)

            for regime in unit
                r += 1
                @test isa(regime, AbstractDrillRegime)
                @test DataDrill(regime) === data

                for lease in regime
                    @test isa(lease, DrillLease)
                    @test DataDrill(lease) === data
                    l += 1

                    for obs in lease
                        t+= 1
                        @test isa(obs, ObservationDrill)
                        @test ichars(obs) == (1.0,)
                        @test _x(obs) == t
                        @test _y(obs) == t
                        @test zchars(obs) == _zchars[t]
                    end

                end
            end
        end
        @test (i,r,l,t) == (1,2,2,5)
    end

    @testset "Create random DataDrill" begin

        theta = [1.0, 1.0, 1.0, 0.5]
        data = DataDrill(TestDrillModel(), theta)

        # go through leases
        j = 0
        t = 0
        for (i,unit) in enumerate(data)
            @test isa(unit, DrillUnit)

            for (r,regimes) in enumerate(unit)
                @test isa(regimes, AbstractDrillRegime)
                @test r in 1:2
                length(regimes) > 0 && (@test sum(j1chars(regimes)) ≈ 1)
                if r == 1
                    @test isa(regimes, DrillInitial)
                else
                    @test isa(regimes, DrillDevelopment)
                end

                for lease in regimes
                    j += 1
                    @test uniti(lease) == i
                    @test isa(lease, DrillLease)

                    for obs in lease
                        t += 1
                    end

                end
            end
        end
        @test j == length(tptr(data))-1
        @test t == length(_y(data))

    end # testset: random drilling data

end # overall testset

end # module
