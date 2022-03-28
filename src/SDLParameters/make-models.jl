export MakeTestData

"Make 2 test datasets: drilling only and the full data (RR + drilling + pdxn)"
function MakeTestData(;
        nz = 15,
        nψ = 13,
        num_i = 50,
        num_zt = 150,
        obs_per_well = 10:20,
        royalty_rates = RoyaltyRatesSmall(),
        reward    = DefaultDrillReward(),
        θ_royalty = Theta(RoyaltyModel()),
        θ_produce = Theta(ProductionModel()),
        θ_drill   = Theta(DefaultDrillReward()),
        wp = FullLeasedProblem(),
        anticipate = true,
        seed=1234,
        ddm_opts = (minmaxleases=1:1, nper_initial=40:40, tstart=1:50)
)

        Random.seed!(seed)

        # grids
        ψs = PsiSpace(nψ)
        u,v = randn(num_i), randn(num_i)
        z = TestPriceProcess(;nsim=num_zt, ngrid=nz)
        _zchars = series(z)
        log_ogip = logOGIP(num_i)

        # Royalty data
        data_roy = TestDataRoyalty(u, v, royalty_rates, θ_royalty, log_ogip)

        # endogenous geology + royalty
        _ichars = SDLParameters.ichars(log_ogip, observed_choices(data_roy))

        ddm   = TestDynamicDrillModel(z, anticipate; reward=reward)
        data_drill = DataDynamicDrill(u, v, _zchars, _ichars, ddm, θ_drill; ddm_opts...)
        data_produce = TestDataProduce(u, data_drill, obs_per_well, θ_produce, log_ogip)

        # big model
        dataset_full  = DataSetofSets(data_drill, data_roy, data_produce, CoefLinks(data_drill))
        theta_full = merge_thetas((θ_drill, θ_royalty, θ_produce), dataset_full)

        rwrd_c = ConstrainedProblem(reward)
        ddm_c = TestDynamicDrillModel(z, anticipate; reward=rwrd_c)
        data_drill_c = DataDrill(data_drill, ddm_c)
        dataset_drill = DataSetofSets(data_drill_c, EmptyDataSet(), EmptyDataSet())
        theta_drill = Theta(rwrd_c)

        return ((dataset_drill, theta_drill, ), (dataset_full, theta_full))
end
