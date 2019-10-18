# look at https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39
# https://discourse.julialang.org/t/two-questions-about-multithreading/14564/2
# https://discourse.julialang.org/t/question-about-multi-threading-performance/12075/3

# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

function check_lengths(ubv::VV, llj::VV, grad::VV, gradj::VM) where {T,VV<:Vector{Vector{T}}, VM<:Vector{Matrix{T}}}
    length(ubv) == length(llj) == length(grad)==length(gradj) || throw(DimensionMismatch())
    all(length.(llj)  .== length(llj[1])  .== size.(gradj, 2)) || throw(DimensionMismatch())
    all(length.(grad) .== length(grad[1]) .== size.(gradj, 1)) || throw(DimensionMismatch())
end

function check_lengths(ubv::VV, llj::VV, grad::VV, gradj::VM) where {T,VV<:Vector{T}, VM<:Matrix{T}}
    size(gradj) == (length(grad), length(llj)) || throw(DimensionMismatch())
end

struct DrillingTmpVars{V<:Vector, VM<:VecOrMat} <: AbstractTmpVars
    ubv::V
    llj::V
    grad::V
    gradJ::VM
    function DrillingTmpVars(ubv::V, llj::V, grad::V, gradJ::VM) where {V,VM}
        # check_lengths(ubv, llj, grad, gradJ)
        return new{V,VM}(ubv, llj, grad, gradJ)
    end
end

const DrillingTmpVarsAll = DrillingTmpVars{<:Vector{<:Vector}}
const DrillingTmpVarsThread = DrillingTmpVars{<:Vector{<:Number}}

_ubv(  dtv::DrillingTmpVars) = dtv.ubv
_llj(  dtv::DrillingTmpVars) = dtv.llj
_grad( dtv::DrillingTmpVars) = dtv.grad
_gradJ(dtv::DrillingTmpVars) = dtv.gradJ

_ubv(  dtv::DrillingTmpVarsAll, id) = getindex( _ubv( dtv), id)
_llj(  dtv::DrillingTmpVarsAll, id) = getindex( _llj( dtv), id)
_grad( dtv::DrillingTmpVarsAll, id) = getindex( _grad(dtv), id)
_gradJ(dtv::DrillingTmpVarsAll, id) = getindex(_gradJ(dtv), id)


function getindex(x::DrillingTmpVars{<:Vector}, i)
    return DrillingTmpVars(_ubv(x,i), _llj(x,i), _grad(x,i), _gradJ(x,i))
end

@noinline function DrillingTmpVars(data::DataDrill, theta::AbstractVector{T}) where {T}
    nth = nthreads()
    J = maxj1length(data)
    maxchoices = num_choices(_model(data))
    k = length(theta)

    ubvs  = Vector{Vector{T}}(undef, nth)
    lljs  = Vector{Vector{T}}(undef, nth)
    grads = Vector{Vector{T}}(undef, nth)
    gradJs = Vector{Matrix{T}}(undef, nth)

    let J=J, maxchoices=maxchoices, k=k, nth=nth
        @threads for id in OneTo(nth)
            tid = T(threadid())
            ubvs[id]  = fill(tid, maxchoices)
            lljs[id]  = fill(tid, J)
            grads[id] = fill(tid, k)
            gradJs[id] = fill(tid, k,J)
        end
    end

    # for i in OneTo(nth)
    #     @assert ubvs[i][1] == i
    #     @assert lljs[i][1] == i
    #     @assert grads[i][1] == i
    #     @assert gradJs[i][1] == i
    # end

    return DrillingTmpVars(ubvs, lljs, grads, gradJs)
end
