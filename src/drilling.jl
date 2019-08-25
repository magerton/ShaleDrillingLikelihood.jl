





function loglik_drilling_history()

    jrng = j1_indexrange(data,i)
    J = length(jrng)

    @views LLmj   = LLJ[:,1:J]
    @views gradmj = gradJ[:,:,1:J]

    fillcols!(LLmj, log.(Prj))
    zero!(gradmj)

    for j in enumerate( jrng )
        for obs in observations(data, j)
            for (m,uv) in enumerate(uvs) # TODO threaded
                LLmj[m,j] += loglik_drilling_in_t(gradmj[:,m,j])
            end
        end
    end

    logsumexp_and_softmax2!(LLm, tmp, LLj)

    gradm[k,m] = sum(gradmj[k,m,:] .* LLj[m,:])

    for obs in observations(data,i)
        for (m,uv) in enumerate(uvs)  # TODO threaded
            LLm[m] += loglik_drilling!(grad,m,uv,drillcomp)
        end # m/ψ
    end # time



end






function loglik_drilling_in_t()

    ubv = view(bigubv, drng.+1, m)

    # payoffs
    @inbounds @simd for d in drng
        ubv[d+1] = flow(u,d,obs) + β * dynamic
    end

    logPrd = ubv[d_obs+1]

    # if no gradient, return
    dograd || return logPrd - logsumexp(ubv)

    logPrd -= logsumexp_and_softmax!(ubv)

    for d in drng
        wt = d==d_obs ? 1-logPrd : -logPrd
        # theta
        for k in idx_drill_cost(model)
            grad[k,m] += wt * (flowdθ(FF, k, θt, σ, wp, s_idx, d, z, ψ, itype...) + prim.β * isev.dEV[z..., ψ, k, sp_idx, itypidx...] )
        end
        # sigma
        if s_idx <= end_ex0(wp)
            dpsi = flowdψ(FF, θt, σ, wp, s_idx, d, z, ψ, itype...) + prim.β * gradient_d(nplus1(Val{NZ}), isev.EV, z..., ψ, sp_idx, itypidx...)::T
            dsig = flowdσ(FF, θt, σ, wp, s_idx, d, z, ψ, itype...) + prim.β * isev.dEVσ[z..., ψ, sp_idx, itypidx...]
            grad[idx_drill_rho(model),m] += wt * (dpsi*_dψ1dθρ(uv..., ρ, σ) + dsig)
        end
    end # d

    return logPrd
end
