module ShaleDrillingLikelihood_DrillingDataStructure_Test

using ShaleDrillingLikelihood
using Test
using Random
using StatsBase
using StatsFuns
using InteractiveUtils
using Dates
using BenchmarkTools

using ShaleDrillingLikelihood: SimulationDraws, _u, _v, SimulationDrawsMatrix, SimulationDrawsVector,
    AbstractDrillModel, DrillModel,
    ExogTimeVars, _timestamp, _timevars, Quarter,
    j1ptr, j2ptr, tptr, zchars, jtstart, ichars, _y, _x, j1chars, hasj1ptr, maxj1length,
    j1start, j1stop, j1length, j1_range,
    tstart, tstop, trange, tlength,
    zcharsvec,
    AbstractDataDrill, DataDrill,
    ObservationGroup,
    Observation, ObservationDrill,
    DrillUnit,
    AbstractRegimeType, InitialDrilling, DevelopmentDrilling, FinishedDrilling,
    AbstractDrillRegime, DrillInitial, DrillDevelopment, DrillLease,
    _i, _data,
    ObservationGroup,
    action, state,
    uniti


function randsumtoone(n)
    x = rand(n)
    x ./= sum(x)
    return x
end

@testset "Drilling Data Structure" begin

    @testset "Exog Time Vars" begin
        nt = 12
        startdate = Date(2003,10)
        daterange = range(startdate; step=Quarter(1), length=nt)
        etv = ExogTimeVars([(x,) for x in randn(nt)], daterange)
        @test (12,1,) == size(etv)
        @test etv[2] == etv[Date(2004,1)]
        @test_throws DomainError etv[Date(2004,2)]
    end

    @testset "initial vs dev't" begin
        @test InitialDrilling()+1 == InitialDrilling()+100 == DevelopmentDrilling()
        @test InitialDrilling()-1 == nothing
        @test DevelopmentDrilling()+1 == FinishedDrilling()
        @test DevelopmentDrilling()-1 == InitialDrilling()
        @test FinishedDrilling()+1 == nothing
        @test FinishedDrilling()-1 == DevelopmentDrilling()
        @test InitialDrilling() in (InitialDrilling(), DevelopmentDrilling())
    end

    @testset "DataDrill Construction" begin
        nt = 12
        _j1ptr   = [1, 1]
        _j2ptr   = 1:1
        _tptr    = [1, 5]
        _jtstart = [1,]
        _jchars  = zeros(Int,0)
        _ichars  = [(1.0,), ]
        y = collect(1:4)
        x = collect(1:4)
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))
        _zchars_short  = ExogTimeVars([(x,) for x in randn(2)], range(Date(2003,10); step=Quarter(1), length=2))

        DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)
        @test_throws DimensionMismatch DataDrill(DrillModel(), [1,], _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)
        @test_throws ErrorException    DataDrill(DrillModel(),       _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars_short)
    end


    @testset "Iteration over DataDrill" begin
        nt = 12
        _j1ptr   = [1, 2]
        _j2ptr   = 2:2
        _tptr    = [1, 5, 6]
        _jtstart = [1, 5]
        _jchars  = [1.0, ]
        _ichars  = [(1.0,), ]
        _tchars  = [(x,x,) for x in 1:5]
        y = collect(1:5)
        x = copy(y)
        _zchars  = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))

        data = DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)

        @test length(data) == 1
        @test maxj1length(data) == 1

        @test j1ptr(data, 1) == 1
        @test j1_range(data,1) == 1:1
        @test j1length(data,1) == 1
        @test j1stop(data,1) == 1
        @test hasj1ptr(data) == true

        @test tstart(data,1) == 1
        @test tstop(data,1) == 4
        @test trange(data,1) == 1:4
        @test tlength(data,1) == 4

        @test tstart(data,2) == 5
        @test tstop(data,2) == 5
        @test trange(data,2) == 5:5
        @test tlength(data,2) == 1

        @test zcharsvec(data, 2) == _timevars(_zchars)[2:end]

        # iterate through all
        i,r,l,t = (0,0,0,0)

        for unit in data
            i+= 1
            @test _i(unit) == i
            @test isa(unit, DrillUnit)
            @test DataDrill(unit) === data
            @test length(unit) == DevelopmentDrilling()

            for regime in unit
                r += 1
                @test isa(regime, AbstractDrillRegime)
                @test DataDrill(regime) === data

                for lease in regime
                    @test isa(lease, DrillLease)
                    @test DataDrill(lease) === data
                    l += 1

                    for obs in lease
                        t+= 1
                        @test isa(obs, ObservationDrill)
                        @test ichars(obs) == (1.0,)
                        @test _x(obs) == t
                        @test _y(obs) == t
                        @test zchars(obs) == _zchars[t]
                    end

                end
            end
        end
        @test (i,r,l,t) == (1,2,2,5)
    end

    @testset "Create random DataDrill" begin

        num_i = 10
        rho = 0.5
        nt = 30
        minmaxleases = 0:3
        nper = 0:10
        _zchars = ExogTimeVars([(x,) for x in randn(nt)], range(Date(2003,10); step=Quarter(1), length=nt))

        psi2 = rand(num_i)
        psi1 = rho .* psi2 .+ (1-rho^2) .* rand(num_i)

        leases_per_unit = [sample(minmaxleases) for i in 1:num_i]

        _jchars = vcat(collect(randsumtoone(lpu) for lpu in leases_per_unit )...)
        num_initial_leases = length(_jchars)
        @test sum(leases_per_unit) == num_initial_leases
        obs_per_lease = vcat(sample(1:10, num_initial_leases), sample(0:10, num_i))
        @test length(obs_per_lease) == num_initial_leases + num_i

        _tptr  = 1 .+ cumsum(vcat(0, obs_per_lease))
        _j1ptr = 1 .+ cumsum(vcat(0, leases_per_unit))
        _j2ptr = (last(_j1ptr)-1) .+ (1:num_i)
        _ichars = [(sample(0:1),) for i in 1:num_i]

        nobs = last(_tptr)-1
        x = sample(1:2, nobs)
        y = zeros(Int, nobs)
        _jtstart = fill(10, num_initial_leases + num_i)

        choice_set = 0:2
        payoffs = zeros(length(choice_set))
        theta = randn(3)

        data = DataDrill(DrillModel(), _j1ptr, _j2ptr, _tptr, _jtstart, _jchars, _ichars, y, x, _zchars)

        pickpsi(a, b, lease::ObservationGroup{<:DrillInitial}) = a
        pickpsi(a, b, lease::ObservationGroup{<:DrillDevelopment}) = b

        function payoff(d::Integer, psi::Real,x::Real,z::Tuple{Real},theta::AbstractVector)
            0 <= d <= 2 || throw(DomainError())
            length(theta)==3 || throw(DimensionMismatch())
            out = d*(theta[1]*psi + theta[2]*x + theta[3]*first(z))
            return Float64(out)
        end

        function simulate_lease(lease)
            nper = length(lease)
            zc = zchars(lease)
            x = _x(lease)
            y = _y(lease)
            @test length(zc) == length(x) == length(y) == nper

            i = uniti(lease)
            psi = pickpsi(psi1[i], psi2[i], lease)

            for t in 1:nper
                f(d) = payoff(d,psi,x[t],zc[t],theta)
                payoffs .=  f.(choice_set)
                @views softmax!(payoffs)
                cumsum!(payoffs, payoffs)
                y[t] = searchsortedfirst(payoffs, rand())-1
            end
        end

        # update leases
        for (i,unit) in enumerate(data)
            for regimes in unit
                for lease in regimes
                    simulate_lease(lease)
                end
            end
        end


    end # testset: random drilling dta

end # overall testset

end # module
