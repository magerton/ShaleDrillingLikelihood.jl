module ShaleDrillingLikelihood_OptimizeDynamic
# using Revise

using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters
using CountPlus, Optim, Test, Distributed, Calculus

using Optim: minimizer
using ShaleDrillingLikelihood: simloglik!, parallel_simloglik!

println("print to keep from blowing up")

# rwrd = DrillReward(
#     DrillingRevenue(Unconstrained(), NoTrend(), GathProcess(), Learn(), WithRoyalty()),
#     DrillingCost_constant(),
#     ExtensionCost_Constant(),
#     ScrapValue_Zero()
# )
# θ_drill = [-6.70387, -1.56538, 1.00494, -2.6653, 0.59405, 0.34476, 0.69659]
# θ_drill = [-6.70387, -1.56538,          -2.6653, 0.59405, 0.34476, 0.69659]

@testset "simulate all data" begin

    DOPAR = true
    TESTGRAD = true

    num_i = 250
    M = 500
    nz = 21
    nψ = 21

    data_theta_pairs = MakeTestData(; num_i=num_i, nz=nz, nψ=nψ) #, reward=rwrd, θ_drill=θ_drill, anticipate=false)

    stopcount!()

    if DOPAR
        rmprocs(getworkers())
        pids = addprocs()
    else
        pids = [1,]
    end

    @everywhere using ShaleDrillingLikelihood

    for (d,t) in data_theta_pairs
        resetcount!()

        theta_peturb = 1.0 .* t

        leo = LocalEstObj(d,theta_peturb)
        reo = RemoteEstObj(leo, M)
        ew = EstimationWrapper(leo, reo)
        leograd = ShaleDrillingLikelihood.grad(leo)

        for dograd in false:true
            @eval @everywhere set_g_RemoteEstObj($reo)
            simloglik!(1, t, dograd, reo)
            # serial_simloglik!(ew, t, dograd)
            parallel_simloglik!(ew, theta_peturb, dograd)
        end

        if TESTGRAD
            g = copy(leograd)
            fdp = Calculus.gradient(x -> parallel_simloglik!(ew, x, false), theta_peturb, :central)
            @test fdp ≈ g
        end

        resetcount!()
        startcount!([50, 500, 100000,], [1, 5, 100,])
        opts = Optim.Options(show_trace=true, time_limit=5*60, allow_f_increases=true)
        res = solve_model(ew, theta_peturb; OptimOpts=opts)
        println(res)
        @test minimizer(res) == theta1(leo)

        println(sprintf_binary(minimizer(res)))
        println(coeftable(leo))
        print("True theta is\n\t")
        println(t)
        print("\n")

        # compare estimate to original coef vector, t
        @test last(Fstat!(leo; H0=t)) == false
    end


    end

end # module
