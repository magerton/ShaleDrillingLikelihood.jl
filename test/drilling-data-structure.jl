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
    ObservationGroup,
    Observation, ObservationDrill,
    DrillUnit,
    AbstractRegimeType, InitialDrilling, DevelopmentDrilling, FinishedDrilling,
    AbstractDrillRegime, DrillInitial, DrillDevelopment, DrillLease,
    _i, _data,
    ObservationGroup,
    action, state


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
        _tchars  = [(x,x,) for x in 1:4]
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))
        _zchars_short  = ExogTimeVars([(x,) for x in randn(2)], range(Date(2003,10); step=Quarter(1), length=2))

        DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)
        @test_throws DimensionMismatch DataDrill(DrillModel(), [1,], _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)
        @test_throws ErrorException    DataDrill(DrillModel(),       _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars_short)
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
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))

        data = DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, _tchars, _zchars)

        @testset "DataDrill" begin
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
                            @test state(obs) == t
                            @test action(obs) == t
                            @test zchars(obs) == _zchars[t]
                        end
                    end
                end
            end
            @test (i,r,l,t) == (1,2,2,5)
        end


        # @testset "DataDrillDevelopment" begin
        #     i,l,t = (0,0,0,)
        #     for history in datadevelopment
        #         i+= 1
        #         @test _i(history) == i
        #
        #         @test length(history) == 1
        #         @test eachindex(history) == 2
        #         @test firstindex(history) == 2
        #         @test lastindex(history) == 2
        #
        #         for lease in history
        #             l += 1
        #             for obs in lease
        #                 t += 1
        #                 @test isa(obs, ObservationDrill)
        #                 @test ichars(obs) == (1.0,)
        #                 @test state(obs) == 4+t
        #                 @test action(obs) == 4+t
        #                 @test zchars(obs) == _zchars[4+t]
        #             end
        #         end
        #     end
        #     @test (i,l,t) == (1,1,1)
        # end
        #
        #
        # @testset "DataDrillInitial" begin
        #     i,l,t = (0,0,0,)
        #     for history in datainitial
        #         i+= 1
        #         @test _i(history) == i
        #
        #         @test isa(history, ObservationGroup{<:DataDrillInitial})
        #         @test isa(history, DrillingHistoryUnit_Initial)
        #         @test isa(history, DrillingHistoryUnit_InitOrDev)
        #         @test isa(_data(history), DataDrillInitial)
        #         @test isa(DataDrill(history), DataDrill)
        #
        #         @test j1length(history) == 1
        #         @test j1_range(history) == 1:1
        #         @test j1start(history) == 1
        #         @test j1stop(history) == 1
        #         @test j1chars(history) == [1.0,]
        #         @test eachindex(history) == 1:1
        #         @test firstindex(history) == 1
        #         @test lastindex(history) == 1
        #
        #         for lease in history
        #             l += 1
        #             @test isa(lease, ObservationGroup{<:DrillingHistoryUnit_Initial})
        #             @test isa(lease, DrillingHistoryLease)
        #             @test isa(_data(lease), DrillingHistoryUnit_Initial)
        #             @test isa(_data(_data(lease)), DataDrillInitial)
        #             @test isa(DataDrill(lease), DataDrill)
        #             @test isa(_data(_data(_data(lease))), DataDrill)
        #             @test length(lease) == 4
        #             @test j1chars(lease) == 1.0
        #             @test eachindex(lease) == 1:4
        #             for obs in lease
        #                 t += 1
        #                 @test isa(obs, ObservationDrill)
        #                 @test ichars(obs) == (1.0,)
        #                 @test state(obs) == t
        #                 @test action(obs) == t
        #                 @test zchars(obs) == _zchars[t]
        #             end
        #         end
        #     end
        #     @test (i,l,t) == (1,1,4)
        # end

    end



end

end # module
