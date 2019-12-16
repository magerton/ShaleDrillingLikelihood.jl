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
    dubv::Matrix{T}
    llj::Vector{T}
    grad::Vector{T}
    gradJ::Matrix{T}
    function DrillingTmpVars(ubv, dubv, llj, grad, gradJ)
        length(grad)==size(gradJ,1) == size(dubv,1) || throw(DimensionMismatch())
        length(ubv) == size(dubv,2) || throw(DimensionMismatch())
        T = eltype(ubv)
        return new{T}(ubv, dubv, llj, grad, gradJ)
    end
end

const DrillingTmpVarsAll{T} = Vector{DrillingTmpVars{T}}
const DrillingTmpVarsThread = DrillingTmpVars

_ubv(  dtv::DrillingTmpVars) = dtv.ubv
_dubv( dtv::DrillingTmpVars) = dtv.dubv
_llj(  dtv::DrillingTmpVars) = dtv.llj
_grad( dtv::DrillingTmpVars) = dtv.grad
_gradJ(dtv::DrillingTmpVars) = dtv.gradJ


function DrillingTmpVars(J::Integer, maxchoices::Integer, k::Integer, T::Type=Float64)
    tid = T(threadid())
    ubv = fill(tid, maxchoices)
    dubv = fill(tid, k, maxchoices)
    llj = fill(tid, J)
    grad = fill(tid, k)
    gradJ = fill(tid, k, J)
    return DrillingTmpVars(ubv, dubv, llj, grad, gradJ)
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
