function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Dates.format),Dates.DateTime,Dates.DateFormat{Symbol("HH:MM:SS   "),Tuple{Dates.DatePart{'H'},Dates.Delim{Char,1},Dates.DatePart{'M'},Dates.Delim{Char,1},Dates.DatePart{'S'},Dates.Delim{String,3}}}})
end
