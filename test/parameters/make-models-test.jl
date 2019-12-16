module ShaleDrillingLikelihood_MakeModelsTest

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Test

@testset "Make Test Datset" begin
    num_i = 25
    (datasm, thetasm, ), (datalg, thetalg,) = MakeTestData(;num_i=num_i)

    @test length.(datasm) == (num_i, 0, 0,)
    @show length.(datalg) == (num_i, num_i, num_i)

end


end
