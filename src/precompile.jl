const SDL = ShaleDrillingLikelihood.SDLParameters

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    revs = (
        DrillingRevenue{Unconstrained,NoTrend,   GathProcess, Learn, WithRoyalty},
        DrillingRevenue{Unconstrained,TimeTrend, GathProcess, Learn, WithRoyalty},
        DrillingRevenue{Unconstrained,TimeFE,    GathProcess, Learn, WithRoyalty},
    )

    costs = (
        DrillingCost_TimeFE,
        DrillingCost_TimeFE_rigrate,
        DrillingCost_TimeFE_costdiffs,
        DrillingCost_TimeFE_rig_costdiffs,
        DrillingCost_DoubleTimeFE,
    )

    for R in revs
        for C in costs
            DR = DrillReward{R,C,ExtensionCost_Constant}
            @assert precompile(Tuple{Type{DataDrillPrimitive}, DR, String})
            @assert precompile(Tuple{Type{DataProduce},        DR, String})
            @assert precompile(Tuple{Type{DataRoyalty},        DR, String})
            @assert precompile(Tuple{Type{DynamicDrillModel},  DR, Float64,LeasedProblem,Tuple{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},UnitRange{Float64}},SparseArrays.SparseMatrixCSC{Float64,Int64},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}},Bool})
            @assert precompile(Tuple{Core.kwftype(typeof(SDL.GridTransition)),NamedTuple{(:minp,),Tuple{Float64}},typeof(SDL.GridTransition),DataDrillPrimitive{DR, ExogTimeVars{2,Tuple{Float64,Float64},StepRange{Dates.Date,Dates.Month}},Tuple{Float64,Float64},Int64,LeasedProblem},Float64,Int64})
        end
    end

    @assert precompile(Tuple{typeof(SDL.parse_commandline)})
    @assert precompile(Tuple{typeof(SDL.PsiSpace),Int64})
    @assert precompile(Tuple{typeof(grad_simloglik!),SubArray{Float64,1,SharedArrays.SharedArray{Float64,2},Tuple{Array{Int64,1},Int64},  false}, ObservationGroup{DataProduce{ProductionModel,Float64},Int64},Array{Float64,1},SimulationDraws{Float64,1,SubArray{Float64,1,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}}})
    @assert precompile(Tuple{typeof(simloglik!),     SubArray{Float64,1,SharedArrays.SharedArray{Float64,2},Tuple{UnitRange{Int64},Int64},true},  ObservationGroup{DataRoyalty{RoyaltyModel,Int64,Array{Float64,1},Float64},Int64},SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true},SimulationDraws{Float64,1,SubArray{Float64,1,Array{Float64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}},Bool})
    @assert precompile(Tuple{typeof(update_reo!),Array{Float64,1}})
    @assert precompile(Tuple{Core.kwftype(typeof(start_up_workers)),NamedTuple{(:nprocs,),Tuple{Int64}},typeof(start_up_workers),Base.EnvDict})

end
