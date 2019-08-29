export flow, loglik_i_thread, loglik_i_serial, cache_pad, irng, loglik, loglik_threaded, loglik_serial

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
        @simd for d in 1:L
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



function loglik_threaded(y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i

    LL = zero(T) # Atomic{T}(zero(T))
    llm = Vector{T}(undef, M)

    ubvs = Vector{Vector{T}}(undef,nthreads())
    init_ubvs(ubvs, L)

    for i in 1:num_i
        fill!(llm, zero(Float64))
        rng = irng(num_t,i)
        xi = view(x, rng)
        yi = view(y, rng)
        @threads for m in 1:M
            local ubv = ubvs[threadid()]
            for t in 1:num_t
                @simd for d in 1:L
                    @inbounds ubv[d] = flow(d, thet, xi[t], psi[m,i], L)
                end
                llm[m] += ubv[yi[t]] - logsumexp(ubv)
            end
        end
        LL += logsumexp(llm) - log(M)
    end
    return LL
end






function loglik_serial(y::AbstractVector, x::AbstractArray, psi::AbstractArray, thet::AbstractVector{T}, num_t::Integer, num_i::Integer) where {T}
    @assert length(y) == length(x) == num_i * num_t
    L = maximum(y)
    M = size(psi,1)
    @assert minimum(y) == 1
    @assert size(psi,2) == num_i

    LL = zero(T) # Atomic{T}(zero(T))

    ubv = Vector{Float64}(undef, L)
    llm = Vector{Float64}(undef, M)

    for i in 1:num_i
        fill!(llm, zero(Float64))
        rng = irng(num_t,i)
        xi = view(x, rng)
        yi = view(y, rng)
        for m in 1:M
            for t in 1:num_t
                @simd for d in 1:L
                    @inbounds ubv[d] = flow(d, thet, xi[t], psi[m,i], L)
                end
                llm[m] += ubv[yi[t]] - logsumexp(ubv)
            end
        end
        LL += logsumexp(llm) - log(M)
    end
    return LL
end
