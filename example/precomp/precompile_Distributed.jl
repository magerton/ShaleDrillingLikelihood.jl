function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Distributed, Symbol("#101#102")) && precompile(Tuple{getfield(Distributed, Symbol("#101#102"))})
    isdefined(Distributed, Symbol("#41#44")) && precompile(Tuple{getfield(Distributed, Symbol("#41#44"))})
    isdefined(Distributed, Symbol("#43#46")) && precompile(Tuple{getfield(Distributed, Symbol("#43#46"))})
    precompile(Tuple{typeof(Distributed.handle_msg),Distributed.JoinCompleteMsg,Distributed.MsgHeader,Sockets.TCPSocket,Sockets.TCPSocket,VersionNumber})
end
