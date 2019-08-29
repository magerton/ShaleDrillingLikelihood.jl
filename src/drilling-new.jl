export flow, loglik_i_thread, loglik_i_serial, cache_pad, irng, loglik

function flow(d::Integer,theta::AbstractVector{T},x::Real,ψ::Real,L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(T)
    return T(theta[d-1]*x + theta[d+1]*ψ)
end

const cache_pad = Int(2)

function loglik_i_thread(yi::AbstractVector{Int}, xi::AbstractVector, psii::AbstractVector, thet::AbstractVector, ubv::AbstractArray, llm::AbstractArray{T}, L::Integer) where {T}
    M = length(psii)
    L = size(ubv,1)
    num_t = length(yi)
    @assert num_t == length(xi)
    @assert length(llm) == M
    @assert size(ubv,2) == nthreads()

    fill!(llm, zero(T))

    @threads for m in 1:M
        for t in 1:num_t
            for d in 1:L
                ubv[d,threadid()] = flow(d, thet, xi[t], psii[m], L)
            end
            d_choice = yi[t]
            @views llm[m] += ubv[d_choice,threadid()] - logsumexp(ubv[:, threadid()])
        end
    end
    return logsumexp(llm)
end



function loglik_i_serial(yi::AbstractVector{Int}, xi::AbstractVector, psii::AbstractVector, thet::AbstractVector, ubv::AbstractArray, llm::AbstractArray{T}, L::Integer) where {T}
    M = length(psii)
    L = size(ubv,1)
    num_t = length(yi)
    @assert num_t == length(xi)
    @assert length(llm) == M
    @assert size(ubv,2) == nthreads()

    fill!(llm, zero(T))

    for m in 1:M
        for t in 1:num_t
            @simd for d in 1:L
                ubv[d,threadid()] = flow(d, thet, xi[t], psii[m], L)
            end
            d_choice = yi[t]
            @views llm[m] += ubv[d_choice,threadid()] - logsumexp(ubv[:, threadid()])
        end
    end
    return logsumexp(llm)
end

irng(num_t::Int,i::Int) = ((i-1)*num_t) .+ 1:num_t

function loglik(f::Function, y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, ubv::AbstractArray, llm::AbstractArray, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    @assert size(ubv,1) == L
    @assert size(ubv,2) == nthreads()
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i

    # @assert length(LLthread) == nthreads()
    # fill!(LLthread, zero(T))

    LL = 0.0 # Atomic{T}(zero(T))
    for i in 1:num_i
        rng = irng(num_t,i)
        @views LL += f(y[rng], x[rng], psi[:,i], thet, ubv, llm, L)
        # LLthread[threadid()] += loglik_i(y[rng], x[rng], psi[:,i], theta, ubv[:,threadid()], llm[:,threadid()], L)
    end
    return LL
end
