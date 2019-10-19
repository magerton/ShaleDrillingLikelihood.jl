# look at https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39
# https://discourse.julialang.org/t/two-questions-about-multithreading/14564/2
# https://discourse.julialang.org/t/question-about-multi-threading-performance/12075/3

# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

# also
# https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142

function check_lengths(ubv::VV, llj::VV, grad::VV, gradj::VM) where {T,VV<:Vector{Vector{T}}, VM<:Vector{Matrix{T}}}
    length(theta) == length(ubv) == length(llj) == length(grad)==length(gradj) || throw(DimensionMismatch())
    all(length.(llj)  .== length(llj[1])  .== size.(gradj, 2)) || throw(DimensionMismatch())
    all(length.(theta) .== length.(grad) .== length(grad[1]) .== size.(gradj, 1)) || throw(DimensionMismatch())
end

function check_lengths(ubv::VV, llj::VV, grad::VV, theta::VV, gradj::VM) where {T,VV<:Vector{T}, VM<:Matrix{T}}
    size(gradj) == (length(grad), length(llj)) || throw(DimensionMismatch())
    length(grad) == length(theta) || throw(DimensionMismatch())
end

struct DrillingTmpVars{V<:Vector, VM<:VecOrMat} <: AbstractTmpVars
    ubv::V
    llj::V
    grad::V
    theta::V
    gradJ::VM
    hess::VM
    gradM::VM
    llm::V
    function DrillingTmpVars(ubv::V, llj::V, grad::V, theta::V, gradJ::VM, hess::VM, gradM::VM, llm::V) where {V,VM}
        # check_lengths(ubv, llj, grad, gradJ)
        return new{V,VM}(ubv, llj, grad, theta, gradJ, hess, gradM, llm)
    end
end

const DrillingTmpVarsAll{T} = DrillingTmpVars{<:Vector{<:Vector{T}}}
const DrillingTmpVarsThread{T} = DrillingTmpVars{<:Vector{T}}


_ubv(  dtv::DrillingTmpVars) = dtv.ubv
_llj(  dtv::DrillingTmpVars) = dtv.llj
_grad( dtv::DrillingTmpVars) = dtv.grad
_gradJ(dtv::DrillingTmpVars) = dtv.gradJ
_theta(dtv::DrillingTmpVars) = dtv.theta
_gradM(dtv::DrillingTmpVars) = dtv.gradM
_hess( dtv::DrillingTmpVars) = dtv.hess
_llm(  dtv::DrillingTmpVars) = dtv.llm

_ubv(  dtv::DrillingTmpVarsAll, id) = getindex( _ubv( dtv), id)
_llj(  dtv::DrillingTmpVarsAll, id) = getindex( _llj( dtv), id)
_grad( dtv::DrillingTmpVarsAll, id) = getindex( _grad(dtv), id)
_gradJ(dtv::DrillingTmpVarsAll, id) = getindex(_gradJ(dtv), id)
_theta(dtv::DrillingTmpVarsAll, id) = getindex(_theta(dtv), id)
_gradM(dtv::DrillingTmpVarsAll, id) = getindex(_gradM(dtv), id)
_hess( dtv::DrillingTmpVarsAll, id) = getindex(_hess( dtv), id)
_llm(  dtv::DrillingTmpVarsAll, id) = getindex(_llm(  dtv), id)

length(dtv::DrillingTmpVarsAll) = length(_ubv(dtv))

function getindex(x::DrillingTmpVars{<:Vector}, i)
    return DrillingTmpVars(
        _ubv(x,i), _llj(x,i), _grad(x,i), _theta(x,i),
        _gradJ(x,i), _hess(x,i), _gradM(x,i), _llm(x,i)
    )
end

@noinline function DrillingTmpVars(data::DataDrill, sim::SimulationDraws{T}) where {T}
    nth = nthreads()
    J = maxj1length(data)
    maxchoices = num_choices(_model(data))
    k,M = size(_drillgradm(sim))

    ubvs  = Vector{Vector{T}}(undef, nth)
    lljs  = Vector{Vector{T}}(undef, nth)
    grads = Vector{Vector{T}}(undef, nth)
    thetas = Vector{Vector{T}}(undef, nth)

    gradJs = Vector{Matrix{T}}(undef, nth)
    hess  = Vector{Matrix{T}}(undef, nth)
    gradM = Vector{Matrix{T}}(undef, nth)
    llm = Vector{Vector{T}}(undef, nth)

    let J=J, maxchoices=maxchoices, k=k, nth=nth
        @threads for id in OneTo(nth)
            tid = T(threadid())
            ubvs[id]  = fill(tid, maxchoices)
            lljs[id]  = fill(tid, J)
            grads[id] = fill(tid, k)
            thetas[id] = fill(tid, k)
            gradJs[id] = fill(tid, k, J)
            hess[id]  = fill(tid, k, k)
            gradM[id] = fill(tid, k, M)
            llm[id]   = fill(tid, M)
        end
    end

    # for i in OneTo(nth)
    #     @assert ubvs[i][1] == i
    #     @assert lljs[i][1] == i
    #     @assert grads[i][1] == i
    #     @assert gradJs[i][1] == i
    # end

    return DrillingTmpVars(ubvs, lljs, grads, thetas, gradJs, hess, gradM, llm)
end

function reset!(dtv::DrillingTmpVarsThread, theta)
    _theta(dtv) .= theta
    fill!(_grad(dtv), 0)
    fill!(_hess(dtv), 0)
end

function reset!(dtv::DrillingTmpVarsAll, theta)
    @threads for i in OneTo(nthreads())
        reset!(dtv[i], theta)
    end
end
