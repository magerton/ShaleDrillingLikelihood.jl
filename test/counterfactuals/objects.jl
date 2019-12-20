module ShaleDrillingLikelihood_CounterfactualObjects_Test

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using Distributed
using Test

using ShaleDrillingLikelihood: drill, _x, _D


num_i = 50
M = 10
nz = 13
nψ = 13

datafull = first(last(MakeTestData(; num_i=num_i, nz=nz, nψ=nψ)))

ddata = drill(datafull)
wpold = statespace(_model(ddata))


newrwrd = DrillReward(
    DrillingRevenue(Constrained(), TimeTrend(), GathProcess(), NoLearn(), WithRoyalty()),
    DrillingCost_TimeFE(2008,2016),
    ExtensionCost_Constant()
)
newwp = PerpetualProblem(wpold)


@testset "Revision of statespace" begin
    newd = DataDrill(ddata, newrwrd, newwp)
    for (xo, xn) in zip(_x(ddata), _x(newd))
        Do = _D(statespace(_model(ddata)), xo)
        Dn = _D(statespace(_model(newd)), xn)
        @test Do == Dn
    end
end

@testset "Creation of data for xfer to workers for simulations" begin
    ddm_novf = DDM_NoVF(_model(ddata))
    data_dso = DataDrillStartOnly(ddata)
    data_for_xfer = DataDrill(data_dso, ddm_novf)
end

end # module
