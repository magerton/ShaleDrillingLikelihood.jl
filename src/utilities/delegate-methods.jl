# copied from https://github.com/JuliaStats/StatsModels.jl/blob/0e59b8ed8c6325175ef9425e89ecab04d87bf70f/src/statsmodel.jl#L1-L33

##############################################################################
#
# A macro for doing delegation
#
# This macro call
#
#     @delegate MyContainer.elems [:size, :length, :ndims, :endof]
#
# produces this block of expressions
#
#     size(a::MyContainer) = size(a.elems)
#     length(a::MyContainer) = length(a.elems)
#     ndims(a::MyContainer) = ndims(a.elems)
#     endof(a::MyContainer) = endof(a.elems)
#
##############################################################################

macro delegate(source, targets)
    typename = esc(source.args[1])
    sourceargs2 = source.args[2]
    fieldname = Expr(:quote, isa(sourceargs2, QuoteNode) ? sourceargs2.value : sourceargs2.args[1])
    funcnames = targets.args
    n = length(funcnames)
    result = quote begin end end
    for i in 1:n
        funcname = esc(funcnames[i])
        f = quote
            ($funcname)(a::($typename), args...; kwargs...) = ($funcname)(getfield(a, $fieldname), args...; kwargs...)
        end
        push!(result.args[2].args, f)
    end
    return result
end
