function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(SharedArrays.finalize_refs),SharedArrays.SharedArray{Float64,2}})
end
