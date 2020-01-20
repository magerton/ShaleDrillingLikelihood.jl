function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(StatsBase.countmap),Array{Float64,1}})
    precompile(Tuple{typeof(StatsBase.countmap),Array{Int32,1}})
end
