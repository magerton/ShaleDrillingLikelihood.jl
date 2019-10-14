module ShaleDrillingLikelihood_DrillingDataStructure_Test

using ShaleDrillingLikelihood
using Test
using Random
using StatsBase
using InteractiveUtils
using Dates
using BenchmarkTools

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    AbstractDrillModel, DrillModel,
    ExogTimeVars, _timestamp, _timevars, Quarter,
    j1ptr, j2ptr, tptr, zchars, jtstart, ichars, tchars, j1chars, hasj1ptr, maxj1length,
    j1start, j1stop, j1length, j1_range,
    tstart, tstop, trange, tlength,
    zcharsvec,
    AbstractDataDrill, DataDrill,
    DataDrillInitial, DataDrillDevelopment,
    AbstractDrillingHistoryUnit,
    DrillingHistoryUnit_Initial,
    DrillingHistoryUnit_Development,
    DrillingHistoryUnit,
    DrillingHistoryUnit_InitOrDev,
    ObservationDrill,
    Observation,
    DrillingHistoryLease,
    _i, _data,
    ObservationGroup

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


    @testset "DataDrill Structure" begin
        nt = 12
        _j1ptr   = [1, 1]
        _j2ptr   = 1:1
        _tptr    = [1, 5]
        _jtstart = [1,]
        _jchars  = zeros(Int,0)
        _ichars  = [(1.0,), ]
        _tchars  = [(x,x,) for x in 1:4]
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))
        _zchars_short  = ExogTimeVars([(x,) for x in randn(2)], range(Date(2003,10); step=Quarter(1), length=2))

        DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)
        @test_throws DimensionMismatch DataDrill(DrillModel(), [1,], _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)
        @test_throws ErrorException    DataDrill(DrillModel(),       _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars_short)
    end

    @testset "DataDrill Structure 2" begin
        nt = 12
        _j1ptr   = [1, 2]
        _j2ptr   = 2:2
        _tptr    = [1, 5, 6]
        _jtstart = [1, 5]
        _jchars  = [1.0, ]
        _ichars  = [(1.0,), ]
        _tchars  = [(x,x,) for x in 1:5]
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))

        data = DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)
        datadevelopment = DataDrillDevelopment(data)
        datainitial = DataDrillInitial(data)

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
        i,l,t = (0,0,0,)
        for history in data
            i+= 1
            @test _i(history) == i
            for lease in history
                l += 1
            end
        end
        @test (i,l) == (1,2)

        @test isa(_data(datainitial), DataDrill)
        @test isa(_data(datadevelopment), DataDrill)

        # iterate through initial drilling
        i,l,t = (0,0,0,)
        for history in datainitial
            i+= 1
            @test _i(history) == i

            @test isa(history, ObservationGroup{<:DataDrillInitial})
            @test isa(history, DrillingHistoryUnit_Initial)
            @test isa(history, DrillingHistoryUnit_InitOrDev)
            @test isa(_data(history), DataDrillInitial)
            @test isa(_data(_data(history)), DataDrill)

            @test j1length(history) == 1
            @test j1_range(history) == 1:1
            @test j1start(history) == 1
            @test j1stop(history) == 1
            @test j1chars(history) == [1.0,]
            @test eachindex(history) == 1:1
            @test firstindex(history) == 1
            @test lastindex(history) == 1

            for lease in history
                l += 1
                @show typeof(lease)
                @test isa(lease, DrillingHistoryUnit_Initial)
                @test isa(_data(lease), DataDrillInitial)
                @test isa(_data(_data(lease)), DataDrill)
                # @test length(lease) == 4
            end
        end
        @test (i,l) == (1,1,)

        # i,l,t = (0,0,0,)
        # for history in datadevelopment
        #     i+= 1
        #     @test _i(history) == i
        #     for lease in history
        #         l += 1
        #     end
        # end
        # @test (i,l) == (1,1,)

    end



end

end # module
