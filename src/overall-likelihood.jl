
function simloglik!(
    grad::AbstractVector,
    grptup::NTuple{N,ObservationGroup}, models::NTuple{N,AbstractModel},
    theta::AbstractVector, sim::SimulationDrawsVector, dograd::Bool
) where {N}

    for (grp,model) in zip(grptup, models)
        subthet = view(theta, models, model)
        simloglik!(obs, model, subthet, sim, dograd)
    end

    if !dograd
        LL = logsumexp(_llm(sim)) - log(_num_sim(sim))
        return LL
    else
        LL = logsumexp_and_softmax!(_llm(sim)) - log(_num_sim(sim))
        for (grp,model) in zip(grptup, models)
            subthet = view(theta, models, model)
            subgrad = view(grad,  models, model)
            grad_simloglik!(subgrad, grp, model, subthet, sim)
        end
        return LL
    end
end


function simloglik!(grad, hess,
    tmpgrads::AbstractMatrix,
    dattup::NTuple{N,AbstractDataSet}, models::NTuple{N,AbstractModel},
    theta::AbstractVector, sim::SimulationDrawsMatrix, dograd::Bool
) where {N}

    LL = 0.0
    # update_ψ1!(sim, theta_royalty_ρ(data,θ))
    # update_dψ1dρ!(sim, theta_royalty_ρ(data,θ))
    # update_nu!(data, model, θ)
    # update_xpnu!(data)


    for i = 1:N
        gradi = view(tmpgrads, :, i)
        grptup = getindex.(dattup, i)
        simi = view(sim, i)
        LL += simloglik!(gradi, grptup, models, theta, simi, dograd)
    end

    if dograd
        mul!(hess, tmpgrads, tmpgrads')
        sum!(reshape(grad, :, 1), grads)
        grad .*= -1
        hess .*= -1
    end
    return -LL
end
