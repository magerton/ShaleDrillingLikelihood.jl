function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Base._mapreduce),typeof(identity),typeof(hcat),IndexLinear,Array{Array{Float64,N} where N,1}})
    precompile(Tuple{typeof(Base.require),Module,Symbol})
end
