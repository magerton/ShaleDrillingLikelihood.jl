function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Type{DataRoyalty},DrillReward{DrillingRevenue{Unconstrained,TimeTrend,GathProcess,Learn,WithRoyalty},DrillingCost_TimeFE,ExtensionCost_Constant},String})
    precompile(Tuple{typeof(ShaleDrillingLikelihood.SDLParameters.parse_commandline)})
end
