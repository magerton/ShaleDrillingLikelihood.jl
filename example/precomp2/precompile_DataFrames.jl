function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Base.Broadcast.materialize!),DataFrames.LazyNewColDataFrame{Symbol},Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1},Nothing,typeof(identity),Tuple{CategoricalArrays.CategoricalArray{Int32,1,UInt32,Int32,CategoricalArrays.CategoricalValue{Int32,UInt32},Union{}}}}})
end
