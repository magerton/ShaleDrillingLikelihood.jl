using Base: OneTo

export flow, irng, loglik_threaded!, loglik_serial!


"""
    logsumexp_and_softmax!(r, x)

Set `r` = softmax(x) and return `logsumexp(x)`.
"""
function logsumexp_and_softmax!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    @assert length(r) == n
    isempty(x) && return -T(Inf)

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

    s = zero(T)
    @inbounds @simd for i in 1:n
        s += ( r[i] = exp(x[i] - u) )
    end
    invs = one(T)/s
    r .*= invs
    return log(s) + u
end

logsumexp_and_softmax!(x) = logsumexp_and_softmax!(x,x)

function flow(d::Integer, theta::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(T)
    return T(theta[d-1]*x + theta[d+1]*ψ)
end

function dflow(k::Integer, d::Integer, thet::AbstractVector{T}, x::Real, ψ::Real, L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    @assert 1 <= k <= length(thet)
    d==1 && return zero(T)
    if d == 2
        k == 1 && return T(x)
        k == 3 && return T(ψ)
        return zero(T)
    else
        k == 2 && return T(x)
        k == 4 && return T(ψ)
        return zero(T)
    end
end

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

irng(num_t::Int,i::Int) = (i-1)*num_t .+ (1:num_t)

function loglik_threaded!(grad::AbstractVector{T}, y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, num_t::Integer, num_i::Integer) where {T}
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

                    @fastmath @inbounds @simd for d in OneTo(L)
                        ubv[d] = flow(d, thet, xi[t], psi[m,i], L)
                    end

                    if !dograd
                        llm[m] += ubv[yi[t]] - logsumexp(ubv)
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



function loglik_serial!(grad::AbstractVector, y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    K = length(thet)
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i
    @assert length(grad) ∈ (0,K)

    dograd = length(grad) > 0

    # LL + grad
    LL = zero(T)
    fill!(grad, zero(T))

    # tmpvars
    ubv = Vector{T}(undef, L)
    llm = Vector{T}(undef, M)
    gradm = Matrix{T}(undef, K, M)

    for i in OneTo(num_i)
        fill!(llm, zero(T))
        dograd && fill!(gradm, zero(T))

        rng = irng(num_t,i)
        xi = view(x, rng)
        yi = view(y, rng)

        for m in OneTo(M)
            for t in OneTo(num_t)

                # LL
                @fastmath @inbounds @simd for d in OneTo(L)
                    ubv[d] = flow(d, thet, xi[t], psi[m,i], L)
                end

                if !dograd
                    llm[m] += ubv[yi[t]] - logsumexp(ubv)
                else
                    llm[m] += ubv[yi[t]] - logsumexp_and_softmax!(ubv)
                    for d in OneTo(L)
                        local wt = T(d == yi[t]) - ubv[d]
                        @fastmath @inbounds @simd for k in OneTo(K)
                            gradm[k,m] += wt * dflow(k, d, thet, xi[t], psi[m,i], L)
                        end # k
                    end # d
                end # if

            end # t
        end  # m

        if !dograd
            LL += logsumexp(llm) - log(M)
        else
            LL += logsumexp_and_softmax!(llm) - log(M)
            grad .+= gradm * llm
        end
    end # i
    return LL
end
