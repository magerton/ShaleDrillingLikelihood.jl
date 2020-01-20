function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base, Symbol("#634#635")) && precompile(Tuple{getfield(Base, Symbol("#634#635"))})
    precompile(Tuple{Core.kwftype(typeof(ArgParse.add_arg_field)),Any,typeof(ArgParse.add_arg_field),ArgParse.ArgParseSettings,Union{Array{T,1}, T} where T<:AbstractString})
end
