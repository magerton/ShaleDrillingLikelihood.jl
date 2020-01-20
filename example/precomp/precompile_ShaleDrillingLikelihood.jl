function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base, Symbol("#722#727")) && precompile(Tuple{getfield(Base, Symbol("#722#727")),Base.RefValue{Any},Tuple{Int64}})
    precompile(Tuple{Core.kwftype(typeof(ShaleDrillingLikelihood.SDLParameters.GridTransition)),NamedTuple{(:minp,),Tuple{Float64}},typeof(ShaleDrillingLikelihood.SDLParameters.GridTransition),DataDrillPrimitive{DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},ExogTimeVars{2,Tuple{Float64,Float64},StepRange{Dates.Date,Dates.Month}},Tuple{Float64,Float64},Int64,LeasedProblem},Float64,Int64})
    precompile(Tuple{Core.kwftype(typeof(ShaleDrillingLikelihood.start_up_workers)),NamedTuple{(:nprocs,),Tuple{Int64}},typeof(start_up_workers),Base.EnvDict})
    precompile(Tuple{Type{DataDrillPrimitive},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DataProduce},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DataRoyalty},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{Type{DynamicDrillModel},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},Float64,LeasedProblem,Tuple{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},UnitRange{Float64}},SparseArrays.SparseMatrixCSC{Float64,Int64},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Bool})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.SDLParameters.PsiSpace),Int64})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.SDLParameters.parse_commandline)})
    precompile(Tuple{typeof(update_reo!),Array{Float64,1}})
end
