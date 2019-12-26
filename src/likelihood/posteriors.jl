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

function SharedPosterior(M,N; kwargs...)
    d = SharedMatrix{Float64}(M,N; kwargs...)
    r = SharedMatrix{Float64}(M,N; kwargs...)
    p = SharedMatrix{Float64}(M,N; kwargs...)
    s = SharedPosterior(d,r,p)
    fill!(s, 0)
    return s
end

SharedPosterior(d::DataFull, M; kwargs...) = SharedPosterior(M, num_i(d); kwargs...)

function view(s::SharedPosterior, i)
    d = view(drill(s), :, i)
    r = view(royalty(s), :, i)
    p = view(produce(s), :, i)
    return (d,r,p)
end

# SML for unit i
function simloglik_posterior!(grptup::NTuple{3,ObservationGroup}, thetas,
    sim::SimulationDrawsVector, posteriors::NTuple{3,AbstractVector};
    kwargs...
) where {N}

    qm = _qm(sim)

    for (k,(theta, grp, post)) in enumerate(zip(thetas, grptup, posteriors))
        fill!(qm, 0)
        simloglik!(theta, grp, theta, sim, false; kwargs...)
        post .= qm
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
    posteriors = get_g_SharedPosterior()
    simloglik_posterior!(i, theta, sim, posteriors::SharedPosterior, data; kwargs...)
    return nothing
end
