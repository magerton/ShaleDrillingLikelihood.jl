export simloglik_posterior!, SharedPosterior


@GenGlobal g_SharedPosterior

struct SharedPosterior
    drill::SharedMatrix{Float64}
    royalty::SharedMatrix{Float64}
    produce::SharedMatrix{Float64}
    function SharedPosterior(d,r,p)
        size(d) == size(r) == size(p) ||
            throw(DimensionMismatch())
        return new(d,r,p)
    end
end

@getFieldFunction SharedPosterior drill royalty produce

function fill!(s::SharedPosterior, x)
    fill!(drill(s), x)
    fill!(royalty(s), x)
    fill!(produce(s), x)
    return nothing
end

function SharedPosterior(M,N)
    d = SharedMatrix{Float64}(M,N)
    r = SharedMatrix{Float64}(M,N)
    p = SharedMatrix{Float64}(M,N)
    s = SharedPosterior(d,r,p)
    fill!(s, 0)
    return s
end

SharedPosterior(d::DataFull, M) = SharedPosterior(M, num_i(d))

function view(s::SharedPosterior, i)
    d = view(drill(s), :, i)
    r = view(royalty(s), :, i)
    p = view(produce(s), :, i)
    return (d,r,p)
end


function parallel_simloglik_posterior!(
    data, theta; kwargs...)
    wp = CachingPool(workers())
    @eval @everywhere

    let x=theta, dograd=dograd, kwargs=kwargs
        pmap(i -> simloglik!(i, x, dograd; kwargs...), wp, OneTo(ew))
    end
    return update!(ew, theta, dograd)
end


# SML for unit i
function simloglik_posterior!(grptup::NTuple{3,ObservationGroup}, thetas,
    sim::SimulationDrawsVector, posteriors::NTuple{3,AbstractVector};
    kwargs...
) where {N}

    for (theta, grp, post) in zip(thetas, grptup, posteriors)
        fill!(_qm(sim), 0)
        simloglik!(theta, grp, theta, sim, false; kwargs...)
        post .= _qm(sim)
    end
    return nothing
end


@noinline function simloglik_posterior!(i, theta, sim, posteriors, data::DataFull; kwargs...)

    grptup = getindex.(data, i)
    simi = view(sim, i)
    ptup = view(posteriors, i)

    θρ = theta_ρ(data, theta)
    update!(simi, θρ)

    thetasvw = split_thetas(data, theta)

    simloglik_posterior!(grptup, thetasvw, simi, ptup; kwargs...)
    return nothing
end


function simloglik_posterior!(i, theta; kwargs...)
    data = get_g_BaseDataSetofSets()
    sim = get_g_SimulationDrawsMatrix()# ::SimulationDraws{Float64,2,Matrix{Float64}}
    posteriors = get_g_SharedPosterior() #::SharedPosterior
    simloglik_posterior!(i, theta, sim, posteriors, data; kwargs...)
    return nothing
end
