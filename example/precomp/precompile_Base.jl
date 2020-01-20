function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Base._compute_eltype),Type{Tuple{Bool,Cmd,Int64}}})
    precompile(Tuple{typeof(Base._mapreduce),typeof(identity),typeof(hcat),IndexLinear,Array{Array{Float64,N} where N,1}})
    precompile(Tuple{typeof(Base.require),Base.PkgId})
    precompile(Tuple{typeof(Base.require),Module,Symbol})
    precompile(Tuple{typeof(Base.require),Module,Symbol})
    precompile(Tuple{typeof(maximum),Array{Int32,1}})
end
