using ShaleDrillingLikelihood: DrillingRevenueTimeTrend,
    DrillingRevenueNoTrend,
    baseyear,
    tech,
    range_i_to_ip1,
    AbstractUnitProblem,
    DateQuarter,
    state_idx,
    InitialDrilling,
    j1chars


export FormulaProduce, FormulaRoyalty

import ShaleDrillingLikelihood: DataRoyalty, DataProduce, DataDrillPrimitive

using StatsBase: countmap
using FileIO: load
using Dates: year
using DataFrames, Query, StatsModels
using Base: OneTo

const cols = Dict(
    :j2ptr   => "j2ptr",
    :j1ptr   => "j1ptr",
    :iwptr   => "iw_ptr",
    :tptr    => "tptr",
    :qptr    => "q_ptr",

    :prices  => "prices",
    :ichars  => "ichars",
    :j1chars => "j1chars",
    :qchars  => "qchars",
    :tchars  => "tchars",
)

# ----------------------------
# Production data
# ----------------------------


# variables for regressions
function FormulaProduce(::DrillReward{<:DrillingRevenueTimeTrend}, startvalues=false)
    strt = @formula(logcumgas_r ~ 0 + intercept + log_ogip + α_t + (1|i) + (1|iD))
    full = @formula(logcumgas_r ~ 0 + intercept + log_ogip + α_t)
    fm = startvalues ? strt : full
    return fm
end
function FormulaProduce(::DrillReward{<:DrillingRevenueNoTrend}, startvalues=false)
    strt = @formula(logcumgas_r ~ 0 + intercept + log_ogip  + (1|i) + (1|iD))
    full = @formula(logcumgas_r ~ 0 + intercept + log_ogip)
    fm = startvalues ? strt : full
    return fm
end
FormulaProduce(x::DynamicDrillModel) = FormulaProduction(reward(x))

function DataProduce(rwrd::DrillReward, path; model=ProductionModel(), kwargs...)

    # load in data
    alldata = load(path)
    _qchars = alldata[cols[:qchars]]
    _qchars[!,:intercept] .= 1
    _qchars[!,:α_t]       .= year.(_qchars[!,:well_start_date]) .- baseyear(rwrd)

    # formula
    startvalues = false
    fm = FormulaProduce(rwrd, startvalues)

    # y, X
    qmf = ModelFrame(fm, _qchars)
    y = response(qmf)
    X = Matrix(modelmatrix(qmf)')

    # data
    obs_ptr   = Int.(alldata[cols[:qptr]])
    group_ptr = Int.(alldata[cols[:iwptr]])
    data = DataProduce(y, X, obs_ptr, group_ptr)
    return data
end

# ----------------------------
# Royalty data
# ----------------------------

FormulaRoyalty(x) = @formula(royalty_id ~ 0 + log_houseval_med + grantor_share_in_state_NO + pct_0_imperv_2001 + log_ogip)

function DataRoyalty(rwrd::DrillReward, path; model=RoyaltyModel(), kwargs...)
    alldata = load(path)
    _ichars = alldata[cols[:ichars]]
    fm = FormulaRoyalty(rwrd)
    rmf = ModelFrame(fm, _ichars)
    y = Int.(response(rmf))
    X = Matrix(modelmatrix(rmf)')

    lhsvar = Symbol(fm.lhs)

    roy_dict = sort(countmap(_ichars[!,:royalty_nearest]))
    royalty_rates = collect(keys(roy_dict))

    # do some checks that data are OK
    y_dict   = sort(countmap(_ichars[!,lhsvar]))
    collect(values(y_dict)) == collect(values(roy_dict)) || throw(error())

    data = DataRoyalty(model, y, X, royalty_rates)
    return data
end

# ----------------------------
# Drilling data primitives
# ----------------------------

itypes(r::DataFrameRow) = ( r[:log_ogip], Int(r[:royalty_id]) )

# ----------------------------
# Prices
# ----------------------------

validnum(x) = isfinite(x) && !ismissing(x)

logprice(  r::DataFrameRow) = log(r[:well_revenue] / r[:ppi2009])
logcost(   r::DataFrameRow) = log(r[:dayrate_c]    / r[:ppi2009])
myyear(    r::DataFrameRow) = Int(floor(r[:qtrdate]))
year_clamp(r::DataFrameRow, lb, ub) = clamp(myyear(r), lb, ub)

function year_clamp(r::DataFrameRow, d::DrillReward{R,<:AbstractDrillingCost_TimeFE}) where {R<:DrillingRevenue}
    c = cost(d)
    return year_clamp(r, ShaleDrillingLikelihood.start(c), ShaleDrillingLikelihood.stop(c))
end

function ztuple(d::DrillReward{R,C}) where {R, C<:Union{DrillingCost_constant, DrillingCost_dgt1}}
    f(r) = (logprice(r), )
    return f
end

function ztuple(d::DrillReward{R,C}) where {R, C<:Union{DrillingCost_TimeFE, DrillingCost_TimeFE_costdiffs}}
    f(r) = (logprice(r), myyear(r))
    return f
end

function ztuple(d::DrillReward{R,C}) where {R, C<:Union{DrillingCost_TimeFE_rigrate, DrillingCost_TimeFE_rig_costdiffs}}
    f(r) = (logprice(r), myyear(r), logcost(r), )
    return f
end

# ----------------------------
# Dates
# ----------------------------

RDate_to_Date(x::Real)::Date = Date(1970,1,1) + Dates.Day(x)

function Ryearmon_to_Date(x::AbstractFloat)
  y = floor(x)
  m = round((x - y) * 12.) + 1
  Date(y,m)
end

position_in_range(x::Real, rng::StepRangeLen) = Int( (x-first(rng)) / step(rng) ) + 1

# ----------------------------
# statespace
# ----------------------------

function (::Type{T})(tc::DataFrame) where {T<:AbstractUnitProblem}
    _dmax  = maximum(tc[!,:d])
    _Dmax  = maximum(tc[!,:D])
    _τ0max = maximum(tc[!,:tau0])
    _τ1max = maximum(tc[!,:tau1])
    _ext   = maximum(tc[tc[!,:tau1] .== 0, :tau0])
    wp_tuple = Int.((_dmax, max(_Dmax,1), _τ0max, _τ1max, _ext))
    println("Making well problem with tuple $(wp_tuple)")
    _wp = T(wp_tuple...)
    return _wp
end

tau1_tau0_D_d1(r::DataFrameRow) = (r[:tau1], r[:tau0], r[:D], r[:d1],)


function sum_to_one!(x::AbstractVector)
    invsumx = 1/sum(x)
    x .*= invsumx
end

# ----------------------------
# Form primitives from datadrill
# ----------------------------

function DataDrillPrimitive(rwrd::DrillReward, path; kwargs...)

    rdata = load(path)
    j1ptr   = Int.(rdata[ cols[:j1ptr] ])
    tptr    = Int.(rdata[ cols[:tptr] ])

    tchars  = rdata[ cols[:tchars] ]
    _j2ptr   = rdata[ cols[:j2ptr] ]
    _j1chars = rdata[ cols[:j1chars] ]
    _ichars  = rdata[ cols[:ichars] ]

    prices  = rdata[ cols[:prices] ] |>
        @filter( validnum(_.dayrate_c) && validnum(_.well_revenue) ) |>
        DataFrame

    z = map(ztuple(rwrd), eachrow(prices))
    minqtr, maxqtr = extrema(prices[!,:qtrdate])
    zrng = DateQuarter(minqtr) : Quarter(1) : DateQuarter(maxqtr)
    qrng = minqtr : 0.25 : maxqtr

    maxt = nrow(tchars)
    searchq(q) = position_in_range(q, qrng)
    jtstart = searchq.(tchars[min.(maxt, tptr[1:end-1]), :qtrdate])

    zchars = ExogTimeVars(z, zrng)
    y = Int.(tchars[!,:d])
    wp = LeasedProblem(tchars)
    x = map(r -> state_idx(wp, tau1_tau0_D_d1(r)...), eachrow(tchars))

    ichars = map(itypes, eachrow(_ichars))
    j2ptr = Int(first(_j2ptr)) : Int(last(_j2ptr))

    @assert length(j2ptr) == length(_j2ptr)

    jwtvec = _j1chars[!, :share_of_unit_lease_owns]

    data = DataDrillPrimitive(
        rwrd, j1ptr, j2ptr, tptr, jtstart,
        jwtvec, ichars, y, x, zchars, wp
    )

    for unit in data
        sum_to_one!(j1chars(unit))
    end

    return data
end
