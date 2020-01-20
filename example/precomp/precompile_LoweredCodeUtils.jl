function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Core.kwftype(typeof(LoweredCodeUtils.methoddef!)),NamedTuple{(:define,),Tuple{Bool}},typeof(LoweredCodeUtils.methoddef!),Function,Array{Any,1},JuliaInterpreter.Frame,Expr,Int64})
end
