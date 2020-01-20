function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(ArgParse.add_arg_field)),Any,typeof(ArgParse.add_arg_field),ArgParse.ArgParseSettings,Union{Array{T,1}, T} where T<:AbstractString})
end
