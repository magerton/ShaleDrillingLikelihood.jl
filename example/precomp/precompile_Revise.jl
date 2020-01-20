function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Revise, Symbol("#32#33")) && precompile(Tuple{getfield(Revise, Symbol("#32#33"))})
    isdefined(Revise, Symbol("#34#35")) && precompile(Tuple{getfield(Revise, Symbol("#34#35"))})
    isdefined(Revise, Symbol("#34#35")) && precompile(Tuple{getfield(Revise, Symbol("#34#35"))})
end
