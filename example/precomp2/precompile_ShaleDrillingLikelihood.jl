function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(ShaleDrillingLikelihood.SDLParameters.GridTransition)),NamedTuple{(:minp,),Tuple{Float64}},typeof(ShaleDrillingLikelihood.SDLParameters.GridTransition),DataDrillPrimitive{DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},ExogTimeVars{2,Tuple{Float64,Float64},StepRange{Dates.Date,Dates.Month}},Tuple{Float64,Float64},Int64,LeasedProblem},Float64,Int64})
    precompile(Tuple{Type{DataDrillPrimitive},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DataProduce},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DataRoyalty},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DynamicDrillModel},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},Float64,LeasedProblem,Tuple{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},UnitRange{Float64}},SparseArrays.SparseMatrixCSC{Float64,Int64},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Bool})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.SDLParameters.parse_commandline)})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.grad_simloglik!),SubArray{Float64,1,SharedArrays.SharedArray{Float64,2},Tuple{Array{Int64,1},Int64},false},ShaleDrillingLikelihood.ObservationGroup{DataProduce{ProductionModel,Float64},Int64},Array{Float64,1},SimulationDraws{Float64,1,SubArray{Float64,1,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}}})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.simloglik!),SubArray{Float64,1,SharedArrays.SharedArray{Float64,2},Tuple{UnitRange{Int64},Int64},true},ShaleDrillingLikelihood.ObservationGroup{DataRoyalty{RoyaltyModel,Int64,Array{Float64,1},Float64},Int64},SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true},SimulationDraws{Float64,1,SubArray{Float64,1,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}},Bool})
end
