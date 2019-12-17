export NormalOGIP, logOGIP,
    PriceProcess, TestPriceProcess,
    SmallLeasedProblem,
    FullLeasedProblem,
    RoyaltyRates,
    RoyaltyRatesSmall,
    RoyaltyRatesFull,
    RoyaltyExogVars,
    RealDiscountRate,
    PsiSpace,
    TestPsiSpace,
    TestDynamicDrillModel,
    TestDataRoyalty,
    TestDataProduce,
    CoefLinks,
    ThetaRho, AlphaPsi, AlphaG, AlphaT, Alpha0,
    Theta,
    DefaultDrillReward,
    BaseYear


# ----------------------------
# Time
# ----------------------------

BaseYear() = ShaleDrillingLikelihood.TIME_TREND_BASE

# ----------------------------
# Geology
# ----------------------------

NormalOGIP(;mu=4.68, sigma=0.31) = Normal(mu,sigma)

logOGIP(num_i, d=NormalOGIP()) = rand(d, num_i)

# ----------------------------
# Price process
# ----------------------------

struct PriceProcess{TS<:TimeSeriesProcess, SRL<:Tuple, ETV<:ExogTimeVars, M<:AbstractMatrix}
    process::TS
    grid::SRL
    series::ETV
    transition::M

    function PriceProcess(process::TS, grid::SRL, series::ETV, transition::M) where {TS,SRL,ETV,M}
        all(minimum.(grid) .<= minimum(series)) || throw(error())
        all(maximum.(grid) .>= maximum(series)) || throw(error())
        length(product(grid...)) == checksquare(transition) || throw(DimensionMismatch())
        new{TS,SRL,ETV,M}(process,grid,series,transition)
    end
end

process(   x::PriceProcess) = x.process
grid(      x::PriceProcess) = x.grid
series(    x::PriceProcess) = x.series
transition(x::PriceProcess) = x.transition
zchars(    x::PriceProcess) = series(x)

function TestAR1Process(;rho=0.8, mean=1.33*(1-rho), var=0.265^2*(1-rho^2))
    return AR1process(mean, rho, var)
end

function TestPriceDates(n=150;start=Date(2003,10), step=Quarter(1))
    return range(start; step=step, length=n)
end

function TestPriceGrid(zvec, length=15; process=TestPriceProcess())
    zmin = min(minimum(zvec), lrmean(process)-3*lrstd(process))
    zmax = max(maximum(zvec), lrmean(process)+3*lrstd(process))
    zrng = range(zmin, zmax; length=length)
    return tuple(zrng)
end

function TestPriceTransition(grid::StepRangeLen; process=TestPriceProcess(), minp=1e-4)
    Pdense = tauchen_1d(process, grid)
    Pdense_zero = zero_out_small_probs(Pdense, minp)
    return sparse(Pdense_zero)
end

function TestPriceProcess(; process=TestAR1Process(), nsim=150, ngrid=13)
    dates = TestPriceDates(nsim)
    series = simulate(process, nsim)
    etv = ExogTimeVars(tuple.(series), dates)
    grid = TestPriceGrid(series, ngrid; process=process)
    transition = TestPriceTransition(grid...; process=process)
    return PriceProcess(process, grid, etv, transition)
end

extend_minmax_factor(x; n) = minimum(x)/n, maximum(x)*n

function BivariateBrownianPriceProcess(etv; extendn=3, np=13, nr=13, )
    p, r, t = Vector.(collect(zip(_timevars(etv)))
    prng = range(extend_minmax_factor)

end


# ----------------------------
# Lease problem
# ----------------------------

SmallLeasedProblem() = LeasedProblem(3, 4, 5, 3, 2)
FullLeasedProblem() = LeasedProblem(8, 8, 12, 12, 8)

# ----------------------------
# Royalty rates
# ----------------------------

RoyaltyRatesSmall() = [1/8, 3/16, 1/4,]
RoyaltyRatesFull() = [1/8, 1/6, 3/16, 1/5, 9/40, 1/4]

function RoyaltyRates(n)
    if n ∈ (3, :small)
        rr = RoyaltyRatesSmall()
    elseif n ∈ (6, :full)
        rr = RoyaltyRatesFull()
    else
        throw(error("don't know this n=$n"))
    end
    return rr::Vector{Float64}
end

function RoyaltyExogVars(num_i::Integer, rates, theta, log_ogip=zeros(num_i,0), model=RoyaltyModel())
    L = length(rates)
    k = length(theta) - (L-1) - extra_parm(model)
    @assert k > 0
    kminus = k - size(log_ogip',1)
    @assert kminus > 0
    Xminus = randn(kminus, num_i)
    X = vcat(log_ogip', Xminus)
    return X
end

# ----------------------------
# Test Data
# ----------------------------

function TestDataRoyalty(u, v, rates, theta, args...)
    num_i = length(u)
    num_i == length(v) || throw(DimensionMismatch())
    Xroyalty = RoyaltyExogVars(num_i, rates, theta, args...)
    data_roy = DataRoyalty(u, v, Xroyalty, theta, rates)
    return data_roy
end

function TestDataProduce(u, datadrill, obs_per_well, theta, log_ogip)
    nwells = total_wells_drilled(datadrill)
    return DataProduce(u, nwells, obs_per_well, theta, log_ogip)
end

# ----------------------------
# Model
# ----------------------------

RealDiscountRate() = ((0x1.006b55c832502p+0)^12 / 1.125) ^ (1/4)
PsiSpace(n; start=-4.5, stop=4.5) = range(start; stop=stop, length=n)
TestPsiSpace() = PsiSpace(13)

ichars(log_ogip, royalty_rates) = collect(zip(log_ogip, royalty_rates))

DefaultDrillReward() = DrillReward(
    DrillingRevenue(Unconstrained(), NoTrend(), NoTaxes() ),
    DrillingCost_constant(),
    ExtensionCost_Constant()
)

function TestDynamicDrillModel(z::PriceProcess, anticipate_t1ev::Bool = true;
    reward = DefaultDrillReward(),
    discount = RealDiscountRate(),
    statespace = FullLeasedProblem(),
    psispace = TestPsiSpace()
)

    zs = grid(z)
    ztrans = transition(z)
    ddm = DynamicDrillModel(
        reward, discount, statespace, zs, ztrans, psispace, anticipate_t1ev
    )
    return ddm
end
