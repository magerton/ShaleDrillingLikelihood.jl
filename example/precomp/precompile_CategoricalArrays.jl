function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Type{CategoricalArrays.CategoricalArray{T,1,V,C,U,U1} where U1 where U where C where V where T},Array{Int32,1}})
    precompile(Tuple{Type{CategoricalArrays.CategoricalArray},Array{Float64,1}})
end
