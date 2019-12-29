using ShaleDrillingLikelihood.SDLParameters
using DataFrames
using Formatting
using TexTables
using JLD2, FileIO
using Dates
using LinearAlgebra
using Statistics
using DataStructures: OrderedDict



using ShaleDrillingLikelihood: idx_ρ, _ρ, _dρdθρ, theta_0, vw_revenue,
    split_theta, revenue,
    theta_produce_σ2η, theta_produce_σ2u, theta_produce_β,
    average_cost_df, theta_indexes


global MGL_CORP_INC_TAX
global CAPITAL_MGL_TAX_RATE


function sigma_epsilon_and_capital_tau(_alpha0, _gamma0, sig_u, sig_eta)
    NOMINAL_ANNUAL_DISCOUNT_RATE = 0.125
    BETA_ANNUAL_NOMINAL = 1/(1+NOMINAL_ANNUAL_DISCOUNT_RATE)
    MONTHLY_INFLATION = 0x1.006b55c832502p+0  #  1.00163780343638
    BETA_ANNUAL_REAL  = (MONTHLY_INFLATION)^12 / (1+NOMINAL_ANNUAL_DISCOUNT_RATE)

    MEDIAN_PERF              = 4428
    MEDIAN_LOGPERF           = log(MEDIAN_PERF)
    GAMMA_Q_T                = 0x1.32694dfd7a02ep+2  #  4.78767728570533    cumulative pdxn FE for pdxn month 240
    GATH_COMP_TRTMT_PER_MCF  = 0.42 + 0.07           # midstream charges
    global MGL_CORP_INC_TAX  = 0.402                 # mgl corp inc tax rate
    IDC_FRACTION             = 0.8                   # fraction of drilling costs that are intangible and expensed
    DEPRECIATION_YEARS       = 7                     # straight-line depreciation for remaining asset over these years
    WELL_COST                = 10                    # in millions of dollars
    MEDIAN_ROYALTY_RATE      = 0.2
    RELATIVE_CAPITAL_MGL_TAX = IDC_FRACTION + (1-IDC_FRACTION) / DEPRECIATION_YEARS * sum( BETA_ANNUAL_REAL.^(1:DEPRECIATION_YEARS) )
    global CAPITAL_MGL_TAX_RATE  = MGL_CORP_INC_TAX * RELATIVE_CAPITAL_MGL_TAX     # 0.3769469 vs 36.8% in HKL (Jan 2019 WP)

    eulers_gamma = Base.MathConstants.eulergamma

    _sigma_epsilon = exp(
        _gamma0 + (sig_u^2 + sig_eta^2)/2 + MEDIAN_LOGPERF + GAMMA_Q_T + log(1-MGL_CORP_INC_TAX) - _alpha0
    )

    return _sigma_epsilon, CAPITAL_MGL_TAX_RATE
end

texify(x::String) = x[1] == '\\' ? ("\$ " * x * " \$") : x

function coef_translate(nm)
    coefdict = Dict(
        "\\theta_\\rho" => "\\rho",
        "\\beta_1"    =>  "Log median house value",
        "\\beta_2"    =>  "Out-of-state owners (share)",
        "\\beta_3"    =>  "Pct impervious",
        "\\beta_4"    =>  "Log OGIP",
        "\\kappa_1"   =>  "0.125 | 0.1667",
        "\\kappa_2"   =>  "0.1667 | 0.1875",
        "\\kappa_3"   =>  "0.1875 | 0.2",
        "\\kappa_4"   =>  "0.2 | 0.225",
        "\\kappa_5"   =>  "0.225 | 0.25",
        "\\gamma_{3}" =>  "Intercept",
    )
    if nm in keys(coefdict)
        nm2 = coefdict[nm]
    else
        nm2 = nm
    end

    return texify(nm2)
end

function nms_coef_se_sumstat(jld2file,
    DATADIR = "E:/projects/haynesville/intermediate_data";
    first_cost_year=FIRST_COST_YEAR, last_cost_year = LAST_COST_YEAR
)

    # retrieve estimates
    println_time_flush("Loading results from $jld2file")
    file = jldopen(jld2file, "r")
        DATAPATH = file["DATAPATH"]
        M        = file["M"]
        ddmnovf  = file["ddm_novf"]
        theta    = file["theta1"]
        LL       = file["LL"]
        g     = file["grad"]
        vcov  = file["invhess"]
    close(file)
    println_time_flush("loaded jld2")

    # info from DDM
    # -------------------

    REWARD = reward(ddmnovf)
    npsi = length(psispace(ddmnovf))
    nz = first(length.(zspace(ddmnovf)))

    # load in data from disk
    # -------------------
    rdatapath = joinpath(DATADIR, DATAPATH)
    println_time_flush("loading $rdatapath")

    data_royalty    = DataRoyalty(REWARD, rdatapath)
    data_produce    = DataProduce(REWARD, rdatapath)
    data_drill_prim = DataDrillPrimitive(REWARD, rdatapath)
    data_drill      = DataDrill(data_drill_prim, ddmnovf)

    # full dataset
    dataset_full = DataSetofSets(data_drill, data_royalty, data_produce, CoefLinks(data_drill))

    idx_d, idx_r, idx_p = theta_indexes(dataset_full)
    theta_d, theta_r, theta_p = split_thetas(dataset_full, theta)

    # make coefs
    # -------------------

    coef = theta
    se = sqrt.(diag(vcov))
    nms = coef_translate.(coefnames(dataset_full))

    # update rho
    # -------------------

    idx_rho = idx_ρ(REWARD)
    thetarho = coef[idx_rho]
    se[idx_rho] *= _dρdθρ(thetarho)
    coef[idx_rho] = _ρ(thetarho)

    # scale
    # -------------------

    theta_rev = vw_revenue(REWARD, theta_d)
    _alpha0 = theta_0(revenue(REWARD), theta_rev)
    _gamma0 = last(theta_produce_β(data_produce, theta_p))
    sig_eta = theta_produce_σ2η(data_produce, theta_p)
    sig_u   = theta_produce_σ2u(data_produce, theta_p)

    sigeps, captau = sigma_epsilon_and_capital_tau(_alpha0, _gamma0, sig_u, sig_eta)

    # costs
    # --------

    cost_df = average_cost_df(dataset_full, theta)
    date_range = first_cost_year .<= year.(cost_df[!,:date]) .<= last_cost_year
    avgcost = mean( cost_df[date_range,:cost2] ) * sigeps / (1-captau)

    sumstats = OrderedDict(
        "\$\\sigma_\\epsilon\$" => sigeps,
        "Avg cost" => -avgcost,
        "Log lik" => LL,
        "Num \$z\$" => nz,
        "Num \$\\psi\$" => npsi,
        "Num simulations" => M,
    )

    idxnew = vcat(idx_r[2:end], idx_d, idx_p[4:end])

    return (nms[idxnew], coef[idxnew], se[idxnew], sumstats)

end

function regcol(title, nms, coefs, ses, sumstats)

    rc = RegCol(title)

    for x in zip(nms, coefs, ses)
        setcoef!(rc, x...)
    end

    setstats!(rc, sumstats)
    rc["Avg cost"].format = "{:.2f}"
    rc["Log lik"].format = "{:.2f}"

    return rc
end



addlinespace(textbl)             = replace(textbl, r"(\\\\)(\n)(?!\s*(Log [Ll]ik|\$ \\sigma\\_\\epsilon \$|Avg cost|Num|\\bottom|&))" => s"\1 \\addlinespace[1pt] \2")
hline_to_midrule(textbl)         = replace(textbl, r"hline" => s"cmidrule(lr)") # cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
header(capt,lab)                 = "\\begin{table}\n\\centering\n\\begin{adjustbox}{max totalheight={\\textheight}, max width={\\textwidth}}\n\\begin{threeparttable}\n\t\\caption{$capt}\\label{$lab}\n\n"
tablenotes(note)                 = "\\begin{tablenotes}[flushleft]\n\\item\\scriptsize{\n\t" * note * "\n}\n\\end{tablenotes}\n"
right_to_left(textbl)            = replace(textbl, r"\\begin\{tabular\}\{r\|([^\}]+)\}" => s"\\begin{tabular}{l\1}")
footer()                         = "\\end{threeparttable}\n\\end{adjustbox}\n\\end{table}"
add_leasing_heading(textable)    = replace(textable, r"(\n +\$ *\\psi\^0 *\$)"   => s" & \\multicolumn{4}{c}{\\emph{Leasing}   } \\\\ \\addlinespace[1pt]\1")
add_drilling_heading(textable)   = replace(textable, r"(\n +\$ *\\alpha\\_\{2008} *\$)" => s" & \\multicolumn{4}{c}{\\emph{Drilling}  } \\\\ \\addlinespace[1pt]\1")
add_production_heading(textable) = replace(textable, r"(\n +Intercept)"          => s" & \\multicolumn{4}{c}{\\emph{Production}} \\\\ \\addlinespace[1pt]\1")


function make_table(rt::IndexedTable, caption::String, label::String, note::String)
    out = header(caption, label) *
    (rt |> to_tex |> addlinespace |> hline_to_midrule |> right_to_left |> add_leasing_heading |> add_drilling_heading |> add_production_heading) *
    tablenotes(note) *footer()
    return out
end
