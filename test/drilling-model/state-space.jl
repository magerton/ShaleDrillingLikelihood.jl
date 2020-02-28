module ShaleDrillingLikelihood_StateSpaceTest

using ShaleDrillingLikelihood
using Test


using ShaleDrillingLikelihood: state_space_vector,
    actionspace,
    state,
    sprime,
    state_if_never_drilled,
    _nstates_per_D,
    end_lrn,
    end_ex0,
    end_ex1,
    _dmax,
    _τrem,
    inf_fm_lrn,
    exploratory_learning,
    ind_lrn,
    maxlease,
    state_space_vector,
    _regime,
    _τ1,
    _τ0,
    _D ,
    _d1,
    ind_exp, ind_inf, end_inf,ind_ex1,ind_ex0,
    state_idx,
    _horizon,
    _nSexp,
    s_of_D,
    post_learning


@testset "Drilling state space" begin

    @testset "sprime of new undrilled" begin
        wp = LeasedProblem(4, 4, 5, 3, 2)
        @test actionspace(wp,11) == 0:0
        @test sprime(wp,11,0) == 11
        @test post_learning(wp,11,0) == 16
    end

    @testset "Maxlease and state_if_never_drilled" begin
        for unitprob ∈ (LeasedProblem, ) # LeasedProblemContsDrill,)
            for wpp in (unitprob(3,4, 5,3,2),) #  unitprob(3,4, 2,3,2), unitprob(3,4, 1,3,5), unitprob(3,16, 5,10,3), unitprob(3,4,1,-1,0))
                SS = state_space_vector(wpp)
                @test maximum(_τrem(s) for s in SS) == maxlease(wpp)
                for sidx ∈ 1:length(wpp)
                    for d in actionspace(wpp,sidx)
                        @test sprime(wpp,sidx,d) ∈ 1:length(wpp)
                    end
                end

                # test state if never drilled...
                for state0 = 1:end_ex0(wpp)
                    statet = state0
                    for t = 1:end_ex0(wpp)
                        statet = sprime(wpp,statet,0)
                        ifno_drill = state_if_never_drilled(wpp,state0,t)
                        @test statet == ifno_drill
                        # println("t,state0 = ($t,$state0): recursive sprime = $statet, state_if_never_drilled = $ifno_drill")
                    end
                end

            end
        end
    end

    @testset "Drilling transitions are ok" begin
        for unitprob ∈ (LeasedProblem, PerpetualProblem, LeasedProblemContsDrill)
            wp = unitprob(4, 4, 5, 3, 2)
            nS = length(wp)

            # infill: d1 == 0 cascades down
            if wp isa LeasedProblemContsDrill
                # infill: d1 == 0
                for i = end_lrn(wp)+1 : 2 : nS-1
                    @test sprime(wp,i,0) == i
                end

                # infill: d1 == 1
                for i = end_lrn(wp)+2 : 2 : nS-2
                    @test sprime(wp,i,0) == i+1
                    for d in 1:_dmax(wp,i)
                        @test sprime(wp,i,d) == i+2*d
                    end
                end

            # Leased or Perpetual
            else
                for i = end_lrn(wp)+1 : nS
                    @test sprime(wp,i,0) == i
                end
            end

            # initial lease term
            if isa(wp,PerpetualProblem)
                @test sprime(wp,1,0) == 1
            else
                for i = 1 :  end_ex1(wp)-1
                    @test sprime(wp,i,0) == i+1
                end

                # last lease term
                for i = end_ex1(wp)+1 : end_ex0(wp)
                    @test sprime(wp,i,0) == i+1
                end

                # any drilling during a primary term
                for i = 0+1 : end_ex0(wp)
                    for d = 1:4
                        @test sprime(wp,i,d) == end_ex0(wp)+1+d
                    end
                end
            end
        end
    end

    @testset "sprime is in 1:length(wp)?" begin
        for unitprob ∈ (LeasedProblem, PerpetualProblem, LeasedProblemContsDrill)
            for dmx in 3:3
                wp = unitprob(dmx, 4, 5, 3, 2)
                SS = state_space_vector(wp)

                @test length(inf_fm_lrn(wp)) == length(exploratory_learning(wp))

                for (i,s) in enumerate(SS)
                    for d in  actionspace(wp,i)
                        sp = sprime(wp,i,d)
                        s = SS[i]  # this state
                        t = SS[sp] # next state
                        sp ∈ 1:length(wp) || println("dmax = $dmx d = $d, i = $i with si = $(SS[i]), sprime = $(sprime(wp,i,d))")
                        @test sp ∈ 1:length(wp)
                        @test t.D == s.D + d

                        SS[i] == state(wp,i) || println("i = $i and d=$d and SS[i]=$(SS[i]) but state(wp,i)=$(state(wp,i))")
                        SS[sp] == state(wp,sp) || println("sp = $sp and d=$d and SS[sp]=$(SS[sp]) but state(wp,sp)=$(state(wp,sp))")

                        if  i < length(wp) && i ∉ ind_lrn(wp) && isa(wp,LeasedProblemContsDrill)
                            @test t.d1 == Int(d > 0)
                        end

                        if i ∈ ind_ex1(wp)
                            @test t.τ1 == ( d == 0 ? s.τ1 -1 : -1)
                            @test t.τ0 == ( d == 0 ? s.τ0 - (i == end_ex1(wp)) : -1)
                        end

                        if i ∈ ind_ex0(wp)
                            @test t.τ1 ==  s.τ1 == -1
                            if isa(wp, PerpetualProblem)
                                @test t.τ0 == -1
                            else
                                @test t.τ0 == ( d == 0 ? s.τ0 - 1 : -1)
                            end
                        end

                    end
                end
            end
        end
    end


    @testset "make State Space" begin

        let dmx = 4,
            Dmx = 4,
            τ0mx = 5,
            τ1mx = 3,
            emx = 2

            for unitprob ∈ (LeasedProblem, PerpetualProblem, LeasedProblemContsDrill)
                wp = unitprob(dmx, Dmx, τ0mx, τ1mx, emx)

                SS = state_space_vector(wp)

                SS[ 1:end_ex1(wp) ]                                # Exploratory 1
                SS[ 1:end_ex0(wp) ]                                # Exploratory 0
                SS[  end_ex0(wp) .+ (1:dmx+1)]                     # Exploratory Terminal + learning update
                # SS[  (τ0mx + τ1mx + dmx+3) .+ (1:2*Dmx-2)]   # Infill drilling
                # SS[  (τ0mx + τ1mx + dmx+3)  +   2*Dmx-2+1]   # Terminal

                nS = length(SS)
                [actionspace(wp,i) for i in 1:nS]

                a = SS, [
                    (i,
                    _regime(wp,i),
                    _τ1(wp,i),
                    _τ0(wp,i),
                    _D(wp,i),
                    _d1(wp,i),
                    collect(sprime(wp,i,d) for d in actionspace(wp,i))
                    ) for i in 1:nS]

                idxs = [ind_exp(wp)..., ind_inf(wp)..., end_inf(wp)..., ind_lrn(wp)...]

                @test idxs ⊆ 1:length(wp)
                @test 1:length(wp) ⊆ idxs

                # test that we get back the state we want to get.
                for i in [ind_exp(wp)..., ind_inf(wp)[1:end-1]..., end_inf(wp)...,]
                    st = SS[i]
                    i_of_st = state_idx(wp, st.τ1, st.τ0, st.D, st.d1)
                    @test i_of_st == i
                    @test i ∈ s_of_D(wp, st.D)
                end

                for s in ind_inf(wp)
                    @test _horizon(wp,s) ∈ (:Infinite, :Finite)
                end

                for s in ind_exp(wp)
                    if isa(wp,PerpetualProblem)
                        @test _horizon(wp,s) == :Infinite
                    else
                        @test _horizon(wp,s) == :Finite
                    end
                end

                for s in ind_lrn(wp)
                    @test _horizon(wp,s) ∈ (:Terminal, :Learning)
                end
            end
        end
    end


    @testset "Checking on dims of dEVσ" begin
        for unitprob ∈ (LeasedProblem, LeasedProblemContsDrill,)
            for wpp in (unitprob(3,4, 5,3,2), unitprob(3,4, 2,3,2), unitprob(3,4, 1,3,5), unitprob(3,16, 5,10,3), unitprob(3,4,1,-1,0))
                nsexp = _nSexp(wpp)
                for sidx ∈ 1:length(wpp)
                    if sidx <= end_ex0(wpp)
                        for d in actionspace(wpp,sidx)
                            @test sprime(wpp,sidx,d) ∈ 1:nsexp
                        end
                    end
                end
            end
        end
    end
end

end
