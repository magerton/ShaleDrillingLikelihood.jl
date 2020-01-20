function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(JLD2.jldopen)),NamedTuple{(:compress,),Tuple{Bool}},typeof(jldopen),Function,String,String})
end
