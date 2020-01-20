function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(StatsModels.collect_matrix_terms),Tuple{StatsModels.InterceptTerm{false},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64}}})
    precompile(Tuple{typeof(copy),Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple},Nothing,typeof(StatsModels.apply_schema),Tuple{Tuple{StatsModels.ConstantTerm{Int64},StatsModels.Term,StatsModels.Term,StatsModels.Term,StatsModels.Term},Base.RefValue{StatsModels.FullRank},Base.RefValue{Type{StatsBase.StatisticalModel}}}}})
    precompile(Tuple{typeof(sum),Tuple{StatsModels.InterceptTerm{false},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64}}})
    precompile(Tuple{typeof(sum),Tuple{StatsModels.InterceptTerm{true},StatsModels.ContinuousTerm{Float64},StatsModels.ContinuousTerm{Float64}}})
end
