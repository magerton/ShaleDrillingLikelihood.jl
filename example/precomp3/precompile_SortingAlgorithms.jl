function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base.Sort, Symbol("##sort!#7")) && precompile(Tuple{getfield(Base.Sort, Symbol("##sort!#7")),SortingAlgorithms.RadixSortAlg,Function,Function,Nothing,Base.Order.ForwardOrdering,typeof(sort!),Array{Int32,1}})
    isdefined(Base.Sort, Symbol("##sort!#7")) && precompile(Tuple{getfield(Base.Sort, Symbol("##sort!#7")),SortingAlgorithms.RadixSortAlg,Function,Function,Nothing,Base.Order.ForwardOrdering,typeof(sort!),Array{Int64,1}})
end
