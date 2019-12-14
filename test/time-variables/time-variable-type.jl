module ShaleDrillingLikelihood_exog_time_var

using ShaleDrillingLikelihood
using Test
using Dates
using SparseArrays

using ShaleDrillingLikelihood: condvar,
    lrvar,
    lrmean,
    simulate,
    zero_out_small_probs,
    approxgrid


@testset "Exog Time Vars: simulation + tauchen" begin
    @testset "Exog Time Vars" begin
        nt = 12
        startdate = Date(2003,10)
        daterange = range(startdate; step=Quarter(1), length=nt)
        etv = ExogTimeVars([(x,) for x in randn(nt)], daterange)
        @test (12,1,) == size(etv)
        @test etv[2] == etv[Date(2004,1)]
        @test_throws DomainError etv[Date(2004,2)]
    end


    @testset "simulate exog time vars and make tauchen" begin

        nt = 120
        startdate = Date(2003,10)
        daterange = range(startdate; step=Quarter(1), length=nt)

        @testset "simulate ar1" begin
            rho = 0.8
            mu = 1.33*(1-rho)
            sigsq = 0.265^2*(1-rho^2)
            price_process = AR1process(mu, rho, sigsq)
            nprice = 51

            let x = price_process
                lrvar(x), lrmean(x), approxgrid(x, nprice)
            end

            prices = simulate(price_process, nt)
            etv = ExogTimeVars(tuple.(prices), daterange)

            for t in 1:nt
                @test tuple(prices[t]) == etv[t]
                @test tuple(prices[t]) == etv[daterange[t]]
            end

            Pprice = tauchen_1d(price_process, nprice)
            Q = zero_out_small_probs(Pprice, 1e-4)
            spPprice = sparse(Q)

            @test all(sum(spPprice; dims=2) .≈ 1)
        end

        @testset "simulate random walk" begin

            # pmin = 0.5574343
            # pmax = 2.1204273
            pmean = 1.3613524
            psd = 0.09380708

            rw = RandomWalkProcess(psd^2)

            prices = simulate(rw, pmean, nt)
            etv = ExogTimeVars(tuple.(prices), daterange)
            pmin, pmax = extrema(prices)

            pgrid = range(pmin/5; stop=pmax*2, length=51)
            Pprice = tauchen_1d(rw, pgrid)
            Q = zero_out_small_probs(Pprice, 1e-5)
            spPprice = sparse(Q)

            @test all(sum(spPprice; dims=2) .≈ 1)

        end

    end
end


end
