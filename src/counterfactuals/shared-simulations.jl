# ----------------------
# Holds simulations
# ----------------------

"Container holds simulations of all sections"
struct SharedSimulations{R<:AbstractRange}
    @addStructFields(SharedMatrix{Float64}, d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub)
    @addStructFields(SharedMatrix{Float64}, epsdeq1, epsdgt1, Prdeq1, Prdgt1)
    @addStructFields(SharedMatrix{Float64}, Eeps, profit, surplus, revenue, drillcost, extension)
    @addStructFields(SharedMatrix{Float64}, D_at_T)
    zchars_time::R
end


"Create `SharedSimulations`"
function SharedSimulations(pids::Vector{<:Integer}, nT::Integer, N::Integer, Dmax::Integer, zchars_time::AbstractRange)

    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub)
    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), epsdeq1, epsdgt1, Prdeq1, Prdgt1)
    @declareVariables( SharedMatrix{Float64}(nT, N, pids=pids), Eeps, profit, surplus, revenue, drillcost, extension)

    D_at_T = SharedMatrix{Float64}(Dmax, N, pids=pids)

    return SharedSimulations(
        d0, d1, d0psi, d1psi, d0eur, d1eur, d0eursq, d1eursq, d0eurcub, d1eurcub,
        epsdeq1, epsdgt1, Prdeq1, Prdgt1,
        Eeps, profit, surplus, revenue, drillcost, extension,
        D_at_T,
        zchars_time
    )
end

function SharedSimulations(pids, data::DataDrill)
    nT = length(zchars(data))
    N = num_i(data)
    Dmax = _Dmax(statespace(_model(data)))
    zchars_time = _timestamp(zchars(data))
    return SharedSimulations(pids, nT, N, Dmax, zchars_time)
end

SharedSimulations(data::DataDrill) = SharedSimulations(workers(), data)

function fill!(s::SharedSimulations, x)
    fill!(s.d0        , x)
    fill!(s.d1        , x)
    fill!(s.d0psi     , x)
    fill!(s.d1psi     , x)
    fill!(s.d0eur     , x)
    fill!(s.d1eur     , x)
    fill!(s.d0eursq   , x)
    fill!(s.d1eursq   , x)
    fill!(s.d0eurcub  , x)
    fill!(s.d1eurcub  , x)
    fill!(s.epsdeq1   , x)
    fill!(s.epsdgt1   , x)
    fill!(s.Prdeq1    , x)
    fill!(s.Prdgt1    , x)
    fill!(s.Eeps      , x)
    fill!(s.profit    , x)
    fill!(s.surplus   , x)
    fill!(s.revenue   , x)
    fill!(s.drillcost , x)
    fill!(s.extension , x)
    fill!(s.D_at_T    , x)
    return nothing
end
