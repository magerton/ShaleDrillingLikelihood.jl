function simloglik!(grad::AbstractVector, grptup::NTuple{2,ObservationGroup}, theta::AbstractVector, sim::SimulationDrawsVector, dograd::Bool)

    fill(_qm(sim), 0)
    logM = log(_num_sim(sim))

    ncoefs = 0
    for grp in grptup
        nparm = _nparm(grp)
        subthet = view(theta, ncoefs .+ OneTo(nparm) )
        ncoefs += nparm
        simloglik!(grp, subthet, sim, dograd)
    end

    if !dograd
        LL = logsumexp(_llm(sim)) - logM
    else
        LL = logsumexp_and_softmax!(_llm(sim)) - logM
        ncoefs = 0
        for grp in grptup
            nparm = _nparm(grp)
            subthet = view(theta, ncoefs .+ OneTo(nparm) )
            subgrad = view(grad,  ncoefs .+ OneTo(nparm) )
            ncoefs += nparm
            grad_simloglik!(subgrad, grp, subthet, sim)
        end
    end
    return LL
end


function simloglik!(grad, hess, tmpgrad,
    dattup::NTuple{2,AbstractDataSet},
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
)

    num_i = length(first(dattup))
    nparm = length(theta)
    (nparm, num_i) == size(tmpgrad) || throw(DimensionMismatch())
    dat_roy = first(dattup)
    dat_pdxn = last(dattup)
    theta_roy =  view(theta, 1:_nparm(first(dattup)))
    theta_pdxn = view(theta, 1+_nparm(first(dattup)) : length(theta))
    ρ = first(theta_roy)

    dtuplens = length.(dattup)
    all(dtuplens .== first(dtuplens)) || throw(DimensionMismatch())

    # for royalty
    update_ψ1!(sim, ρ)
    update_dψ1dρ!(sim, ρ)
    update_xbeta!(dat_roy, theta_royalty_β(dat_roy, theta_roy))

    # for pdxn
    update_xsum!(dat_pdxn)
    update_nu!(dat_pdxn, theta_pdxn)
    update_xpnu!(dat_pdxn)

    LL = 0.0
    fill!(tmpgrad, 0)

    for i = 1:num_i
        gradi = view(tmpgrad, :, i)
        grptup = getindex.(dattup, i)
        simi = view(sim, i)
        fill!(_qm(sim), 0)
        LL += simloglik!(gradi, grptup, theta, simi, dograd)
    end

    if dograd
        mul!(hess, tmpgrad, tmpgrad')
        sum!(reshape(grad, :, 1), tmpgrad)
        grad .*= -1
        hess .*= -1
    end

    return -LL
end
