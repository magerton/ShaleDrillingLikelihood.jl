using ShaleDrillingLikelihood: DrillingRevenueTimeTrend,
    DrillingRevenueNoTrend,
    DrillingRevenueTimeFE,
    baseyear,
    tech,
    range_i_to_ip1,
    AbstractUnitProblem,
    DateQuarter,
    state_idx,
    InitialDrilling,
    j1chars,
    yearrange


export FormulaProduce, FormulaRoyalty, ThetaProduceStarting

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
    strt = @formula(logcumgas_r ~ 1 + log_ogip + α_t + (1|icat) + (1|iD))
    full = @formula(logcumgas_r ~ 1 + log_ogip + α_t)
    fm = startvalues ? strt : full
    return fm
end
function FormulaProduce(::DrillReward{<:DrillingRevenueNoTrend}, startvalues=false)
    strt = @formula(logcumgas_r ~ 1 + log_ogip  + (1|icat) + (1|iD))
    full = @formula(logcumgas_r ~ 1 + log_ogip)
    fm = startvalues ? strt : full
    return fm
end
function FormulaProduce(::DrillReward{<:DrillingRevenueTimeFE}, startvalues=false)
    strt = @formula(logcumgas_r ~ 1 + log_ogip + well_start_fe  + (1|icat) + (1|iD))
    full = @formula(logcumgas_r ~ 1 + log_ogip + well_start_fe)
    fm = startvalues ? strt : full
    return fm
end

FormulaProduce(x::DynamicDrillModel) = FormulaProduction(reward(x))


function yearmonth_float(t)
    ym = yearmonth(t)
    return first(ym) + (last(ym)-1)/12
end

function ym_to_ur(ur::UnitRange, ym::Float64)
    baseyr = first(ur)
    lastyr = last(ur)
    y = floor(ym - baseyr) + baseyr
    return clamp(y, baseyr, lastyr)
end

ym_to_ur(ur::UnitRange, t::Date) = ym_to_ur(ur, yearmonth_float(t))
ym_to_ur(rwrd::DrillReward, x) = ym_to_ur(yearrange(tech(revenue(rwrd))), x)


function update_qchars!(_qchars::DataFrame, rwrd)
    _qchars[!,:α_t]  .= year.(_qchars[!,:well_start_date]) .- baseyear(rwrd)
    _qchars[!,:icat] .= CategoricalVector(_qchars[!,:i])
    wsy = map(t -> ym_to_ur(rwrd, t), _qchars[!,:well_start_date])
    _qchars[!,:well_start_fe] = CategoricalArray(wsy)
    return _qchars
end


function DataProduce(rwrd::DrillReward, path; model=ProductionModel(), kwargs...)

    # load in data
    alldata = load(path)
    _qchars = alldata[cols[:qchars]]
    update_qchars!(_qchars,rwrd)

    # formula
    startvalues = false
    fm = FormulaProduce(rwrd, startvalues)

    # y, X
    qmf = ModelFrame(fm, _qchars)
    y = response(qmf)
    X = Matrix(modelmatrix(qmf)')
    X[1:end-1,:] .= X[2:end,:]
    X[end,:] .= 1

    @assert all(@view(X[end,:]) .== 1)

    # data
    obs_ptr   = Int.(alldata[cols[:qptr]])
    group_ptr = Int.(alldata[cols[:iwptr]])
    data = DataProduce(y, X, obs_ptr, group_ptr)
    return data
end

function ThetaProduceStarting(rwrd::DrillReward, path)

    alldata = load(path)
    _qchars = alldata[cols[:qchars]]
    update_qchars!(_qchars,rwrd)

    startingvals = true
    fm = FormulaProduce(rwrd, startingvals)
    res = fit!(LinearMixedModel(fm, _qchars))
    σwell, σψ, σϵ = vcat(std(res)...)
    cf = coef(res)
    qcoef = vcat(σψ, cf[2:end], cf[1], σϵ, σwell)
    return qcoef
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

itypes(r::DataFrameRow) = ( r[:log_ogip], r[:royalty_nearest], )

# ----------------------------
# Prices
# ----------------------------

validnum(x) = isfinite(x) && !ismissing(x)

logprice(  r::DataFrameRow) = log(r[:well_revenue] / r[:ppi2009])
logrigrate(r::DataFrameRow) = log(r[:dayrate_c]    / r[:ppi2009])
myyear(    r::DataFrameRow, baseyr=BaseYear()) = floor(r[:qtrdate] - baseyr) + baseyr
year_clamp(r::DataFrameRow, lb, ub) = clamp(myyear(r), lb, ub)

@deprecate logcost(r) logrigrate(r)

function year_clamp(r::DataFrameRow, d::DrillReward{R,<:AbstractDrillingCost_TimeFE}) where {R<:DrillingRevenue}
    c = cost(d)
    return year_clamp(r, ShaleDrillingLikelihood.start(c), ShaleDrillingLikelihood.stop(c))
end

function ztuple(d::DrillReward{R,C}) where {R, C<:Union{DrillingCost_constant, DrillingCost_dgt1}}
    f(r) = (logprice(r), )
    return f
end

function ztuple(d::DrillReward{R,C}) where {R, C<:AbstractDrillingCost_TimeFE}
    f(r) = (logprice(r), myyear(r))
    return f
end

function ztuple(d::DrillReward{R,C}) where {R, C<:Union{DrillingCost_TimeFE_rigrate, DrillingCost_TimeFE_rig_costdiffs}}
    f(r) = (logprice(r), logrigrate(r), myyear(r), )
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
    _qchars  = rdata[ cols[:qchars] ]

    last_well_drill_year = year(maximum(_qchars[!,:well_start_date]))
    first_tchars, last_tchars = extrema(tchars[!,:qtrdate])
    last_year = max(last_well_drill_year, floor(last_tchars))

    prices  = rdata[ cols[:prices] ] |>
        @filter(
            validnum(_.dayrate_c) &&
            validnum(_.well_revenue) &&
            floor(first_tchars) <= _.qtrdate &&
            _.qtrdate < last_year+1
        ) |> DataFrame

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

    if length(j1ptr) == 0
        newj1ptr = ones(length(ichars)+1)
    else
        newj1ptr = j1ptr
    end

    data = DataDrillPrimitive(
        rwrd, newj1ptr, j2ptr, tptr, jtstart,
        jwtvec, ichars, y, x, zchars, wp
    )

    for unit in data
        sum_to_one!(j1chars(unit))
    end

    return data
end
