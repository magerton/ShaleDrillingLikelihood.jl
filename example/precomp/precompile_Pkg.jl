function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Pkg.Artifacts.do_artifact_str),String,Dict{String,Any},String,Module})
    precompile(Tuple{typeof(Pkg.Artifacts.do_artifact_str),String,Dict{String,Any},String,Module})
end
