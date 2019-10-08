using Base: OneTo

export flow, simloglik_drill!

# drilling.jl
irng(num_t::Int, i::Int) = (i-1)*num_t .+ (1:num_t)

# look at https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
# https://discourse.julialang.org/t/poor-performance-on-cluster-multithreading/12248/39
# https://discourse.julialang.org/t/two-questions-about-multithreading/14564/2
# https://discourse.julialang.org/t/question-about-multi-threading-performance/12075/3

# check out this??
# https://discourse.julialang.org/t/anyone-developing-multinomial-logistic-regression/23222

@noinline function init_ubvs(ubvs::Vector{Vector{T}}, L::Integer) where {T}
    # Allocate the tmpvars the thread's own heap lazily
    # instead of the master thread heap to minimize memory conflict.
    # https://github.com/nacs-lab/yyc-data/blob/d082032d075070b133fe909c724ecc405e80526a/lib/NaCsCalc/src/utils.jl#L120-L142
    @threads for id in 1:nthreads()
        tid = threadid()
        ubvs[tid] = fill(T(tid), L)
    end
end


function simloglik_drill!(grad::AbstractVector{T}, y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    K = length(thet)
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i
    @assert length(grad) ∈ (0,K)

    dograd = length(grad) > 0

    # for out
    LL = zero(T)
    fill!(grad, zero(T))

    # tmpvar for sims
    llm = Vector{T}(undef, M)
    gradm = Matrix{T}(undef, K, M)

    # tmpvar for threads
    ubvs = Vector{Vector{T}}(undef,nthreads())
    gradtmps = Vector{Vector{T}}(undef,nthreads())
    init_ubvs(ubvs, L)
    init_ubvs(gradtmps, K)

    for i in OneTo(num_i)
        fill!(llm, zero(Float64))

        rng = irng(num_t,i)
        xi = view(x, rng)
        yi = view(y, rng)
        let M=M, num_t=num_t, L=L, K=K, dograd=dograd, ubvs=ubvs, gradtmps=gradtmps, thet=thet, xi=xi, psi=psi, llm=llm, gradm=gradm
            @threads for m in OneTo(M)
                local ubv = ubvs[threadid()]
                local gradtmp = gradtmps[threadid()]
                dograd && fill!(gradtmp, zero(T))

                for t in OneTo(num_t)

                    # loop over choices
                    @fastmath @inbounds @simd for d in OneTo(L)
                        ubv[d] = flow(d, thet, xi[t], psi[m,i], L)
                    end

                    # compute likliehood
                    if !dograd
                        llm[m] += ubv[yi[t]] - logsumexp(ubv)

                    # compute liklihood + gradient
                    else
                        llm[m] += ubv[yi[t]] - logsumexp_and_softmax!(ubv)
                        for d in OneTo(L)
                            local wt = T(d == yi[t]) - ubv[d]
                            @fastmath @inbounds @simd for k in OneTo(K)
                                gradtmp[k] += wt * dflow(k, d, thet, xi[t], psi[m,i], L)
                            end # k
                        end # d
                    end # if

                end # t
                dograd && (gradm[:,m] .= gradtmp)
            end # m
        end # let

        if !dograd
            LL += logsumexp(llm) - log(M)
        else
            LL += logsumexp_and_softmax!(llm) - log(M)
            mul!(gradtmps[1], gradm, llm)
            grad .+= gradtmps[1]
        end

    end # i

    return LL

end
