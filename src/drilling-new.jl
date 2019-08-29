export flow, loglik_i_thread, loglik_i_serial, cache_pad, irng, loglik

function flow(d::Integer,theta::AbstractVector{T},x::Real,ψ::Real,L::Integer) where {T}
    @assert 1 <= d <= L  # FIXME with L
    d==1 && return zero(T)
    return T(theta[d-1]*x + theta[d+1]*ψ)
end

const cache_pad = Int(100)

function do_assertions(yi::AbstractVector{Int}, xi::AbstractVector, psii::AbstractVector, thet::AbstractVector, ubv::AbstractArray, llm::AbstractArray{T}, L::Integer) where {T}
    M = length(psii)
    num_t = length(yi)
    @assert num_t == length(xi)
    @assert length(llm) == M
    @assert size(ubv,2) == nthreads()
    return M, num_t
end

function inner_loop!(llmview, yi, xi, psii_m, thet, ubvview)
    num_t = length(yi)
    L = length(ubvview)

    for t in 1:num_t
        for d in 1:L
            @inbounds ubvview[d] = flow(d, thet, xi[t], psii_m, L)
        end
        @views llmview[1] += ubvview[ yi[t] ] - logsumexp(ubvview)
    end

end

function loglik_i_thread(yi::AbstractVector{Int}, xi::AbstractVector, psii::AbstractVector, thet::AbstractVector, ubv::AbstractArray, llm::AbstractArray{T}, L::Integer) where {T}
    M, num_t = do_assertions(yi,xi,psii,thet,ubv,llm,L)
    fill!(llm, zero(T))
    @threads for m in 1:M
        inner_loop!( view(llm, m), yi, xi, psii[m], thet, view(ubv, 1:L, threadid()))
        # inner_loop!( view(llm, m), yi, xi, psii[m], thet, Vector{Float64}(undef, L))
    end
    return logsumexp(llm) - log(M)
end


function loglik_i_serial(yi::AbstractVector{Int}, xi::AbstractVector, psii::AbstractVector, thet::AbstractVector, ubv::AbstractArray, llm::AbstractArray{T}, L::Integer) where {T}
    M, num_t = do_assertions(yi,xi,psii,thet,ubv,llm,L)
    fill!(llm, zero(T))
    for m in 1:M
        inner_loop!( view(llm, m), yi, xi, psii[m], thet, view(ubv, 1:L, threadid()))
    end
    return logsumexp(llm) - log(M)
end

irng(num_t::Int,i::Int) = (i-1)*num_t .+ (1:num_t)

function loglik(f::Function, y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, ubv::AbstractArray, llm::AbstractArray, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    @assert size(ubv,1) == L+cache_pad
    @assert size(ubv,2) == nthreads()
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i

    LL = zero(T) # Atomic{T}(zero(T))
    for i in 1:num_i
        rng = irng(num_t,i)
        @views LL += f(y[rng], x[rng], psi[:,i], thet, ubv, llm, L)
        # LLthread[threadid()] += loglik_i(y[rng], x[rng], psi[:,i], theta, ubv[:,threadid()], llm[:,threadid()], L)
    end
    return LL
end
