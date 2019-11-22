# look at https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39
# https://discourse.julialang.org/t/two-questions-about-multithreading/14564/2
# https://discourse.julialang.org/t/question-about-multi-threading-performance/12075/3

# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

# also
# https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142

struct DrillingTmpVars{T<:AbstractFloat} <: AbstractTmpVars
    ubv::Vector{T}
    llj::Vector{T}
    grad::Vector{T}
    theta::Vector{T}
    gradJ::Matrix{T}
    function DrillingTmpVars(ubv, llj, grad, theta, gradJ)
        length(theta) == length(grad)==size(gradJ,1) || throw(DimensionMismatch())
        T = eltype(ubv)
        return new{T}(ubv, llj, grad, theta, gradJ)
    end
end

const DrillingTmpVarsAll{T} = Vector{DrillingTmpVars{T}}
const DrillingTmpVarsThread = DrillingTmpVars

_ubv(  dtv::DrillingTmpVars) = dtv.ubv
_llj(  dtv::DrillingTmpVars) = dtv.llj
_grad( dtv::DrillingTmpVars) = dtv.grad
_gradJ(dtv::DrillingTmpVars) = dtv.gradJ
_theta(dtv::DrillingTmpVars) = dtv.theta

# @deprecate _ubv(  dtv::DrillingTmpVarsAll, id) _ubv(  dtv[id])
# @deprecate _llj(  dtv::DrillingTmpVarsAll, id) _llj(  dtv[id])
# @deprecate _grad( dtv::DrillingTmpVarsAll, id) _grad( dtv[id])
# @deprecate _gradJ(dtv::DrillingTmpVarsAll, id) _gradJ(dtv[id])
# @deprecate _theta(dtv::DrillingTmpVarsAll, id) _theta(dtv[id])

function DrillingTmpVars(J::Integer, maxchoices::Integer, k::Integer, T::Type=Float64)
    tid = T(threadid())
    ubv = fill(tid, maxchoices)
    llj = fill(tid, J)
    grad = fill(tid, k)
    theta = fill(tid, k)
    gradJ = fill(tid, k, J)
    return DrillingTmpVars(ubv, llj, grad, theta, gradJ)
end

@noinline function DrillingTmpVars(J::Integer, model::AbstractDrillModel, T::Type=Float64)
    nth = nthreads()
    maxchoices = num_choices(model)
    k = _nparm(model)

    dtvs = Vector{DrillingTmpVars{T}}(undef, nth)

    let J=J, maxchoices=maxchoices, k=k
        @threads for id in OneTo(nth)
            dtvs[id] = DrillingTmpVars(J, maxchoices, k, T)
        end
    end

    return dtvs
end

update_theta!(dtv::DrillingTmpVarsThread, theta) = (_theta(dtv) .= theta)

function update_theta!(dtv::DrillingTmpVarsAll, theta)
    @threads for i in OneTo(nthreads())
        update_theta!(dtv[i], theta)
    end
end
