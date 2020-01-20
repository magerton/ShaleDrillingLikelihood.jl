function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(RData.fileio_load),FileIO.File{FileIO.DataFormat{:RData}}})
    precompile(Tuple{typeof(RData.jlvec),RData.RVector{Float64,0x0e},Bool})
    precompile(Tuple{typeof(RData.sexp2julia),RData.RVector{Int32,0x0d}})
end
