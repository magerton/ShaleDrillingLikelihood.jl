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
    ObservationDrill, DataDrill, ObservationGroupDrill, DataOrObsDrill, AbstractDataStructureDrill,
    ExogTimeVars, _timestamp, _timevars, Quarter


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

end

end # module
