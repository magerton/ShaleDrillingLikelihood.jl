module ShaleDrillingLikelihood_check_psi

using ShaleDrillingLikelihood
using Test
using SparseArrays
using Random
using StatsFuns

using Base.Iterators: product, OneTo

using ShaleDrillingLikelihood: DCDPEmax,
    DCDPTmpVars,
    flow!,
    end_ex0,
    reward,
    revenue,
    _dψ1dρ,
    _dψ1dθρ,
    _ψ1,
    _ρ,
    dp1space,
    sprimes,
    SimulationDraws,
    SimulationDraw,
    _psi1, _psi2, _ψ1, _ψ2


println("print to keep from blowing up")

@testset "DDM - thetas" begin

    # set up coefs
    θρ = 0.0
    αψ = 1.0 # 0.33
    αg = 0.56

    θ_drill_u = [-6.5, -0.85, -2.8, αg, αψ, θρ]
    θ_drill_c = vcat(θ_drill_u[1:3], θρ)

    Random.seed!(1234)
    f = DrillReward(DrillingRevenue(Constrained(),NoTrend(),NoTaxes()), DrillingCost_constant(), ExtensionCost_Constant())
    theta = randn(_nparm(f))
    # @show theta

    nψ, dmx, nz =  13, 3, 11

    # psi
    ψs = range(-4.5; stop=4.5, length=nψ)

    # z transition
    zs = ( range(-1.5; stop=1.5, length=nz), )
    nz = prod(length.(zs))
    ztrans = spdiagm(-1 => fill(1.0/3, nz-1), 0 => fill(1.0/3, nz), 1 => fill(1.0/3, nz-1) )
    ztrans[1,1] = ztrans[1,2] = ztrans[end,end-1] = ztrans[end,end] = 0.5

    # wp = LeasedProblemContsDrill(dmx,4,5,3,2)
    wp = LeasedProblem(dmx,4,5,3,2)

    # ichars
    ichar = (2.0, 0.25,)

    # ddm object
    ddm = DynamicDrillingModel(f, 0.9, wp, zs, ztrans, ψs, false)
    u,v = (0.3, -1.5)
    θρ = 0.5
    rho = _ρ(θρ)
    @test rho == logistic(θρ)
    sim = SimulationDraw(u,v, θρ)
    @test _psi1(sim) == _ψ1(sim)
    @test _psi2(sim) == _ψ2(sim)
    @test u == _psi2(sim) == _ψ2(u,v)
    @test (rho*u + sqrt(1-rho^2)*v) == _psi1(sim) == _ψ1(sim)

    for sidx in 1:length(wp)
        D = _D(wp,sidx)
        
        # @inline _ψ(    m::AbstractDrillModel, state, s::SimulationDraw) = _Dgt0(m, state) ? _ψ2(s) : _ψ1(s)
        # @inline _dψdθρ(m::AbstractDrillModel, state, s::SimulationDraw) = _Dgt0(m, state) ? azero(s) : _dψ1dθρ(s)

    end

end # testset

end # module
