# Overall structure
#----------------------------

import ArgParse: parse_item
export parse_commandline, print_parsed_args

function arg_settings()
    s = ArgParseSettings()

    add_arg_group(s, "Computation", "comp")
    add_arg_group(s, "Payoffs", "payoff")
    add_arg_group(s, "Approximation", "approx")
    add_arg_group(s, "Parameters", "param")

    @add_arg_table s begin

        # Dataset
        "--dataset", "-d"
            help = "Dataset to use"
            arg_type = String
            default = "data_all_leases.RData"

        # Computation control
        "--Mcnstr"
            arg_type = Int
            default = 500
            group = "comp"
        "--Mfull"
            arg_type = Int
            default  = 2000
            group = "comp"

        "--doCnstr"
            action = :store_true
            group = "comp"
        "--noFull"
            action = :store_true
            group = "comp"

        "--maxtimeCnstr"
            arg_type = Int
            default = 0 * 60^2
            group = "comp"
        "--maxtimeFull"
            arg_type = Int
            default  = 0 * 60^2
            group = "comp"
        "--noPar"
            action = :store_true
            group = "comp"

        # Parameters
        "--computeStarting"
            action = :store_true
            group = "param"
        "--theta"
            arg_type = Vector{Float64}
            default = zeros(0)
            group = "param"
        "--convertKappa"
            action = :store_true
            group = "param"

        # Payoffs
        "--cost"
            arg_type = AbstractDrillingCost
            default = DrillingCost_TimeFE(2008.5,2012.5)
            group = "payoff"
        "--extension"
            arg_type = AbstractExtensionCost
            default = ExtensionCost_Constant()
            group = "payoff"
        "--revenue"
            arg_type = DrillingRevenue
            default = DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() )
            # default = DrillingRevenue(Unconstrained(), TimeFE(2008.5,2016.5), GathProcess() )
            group = "payoff"

        "--anticipateT1EV"
            help = "T1ev shocks anticipated"
            action = :store_true
            group = "payoff"
        "--discount"
            arg_type = Float64
            default = RealDiscountRate()
            group = "payoff"

        # Approximation
        "--numP"
            arg_type=Int
            default=13
            group="approx"
        "--numPsi"
            arg_type=Int
            default=13
            group="approx"
        "--extendPriceGrid"
            arg_type=Float64
            default=log(2)
            group="approx"
        "--minTransProb"
            arg_type=Float64
            default=minp_default()
            group="approx"
    end
    return s
end

parse_commandline() = parse_args(arg_settings())

ArgParse.parse_item(::Type{AbstractDrillingCost}, x::AbstractString) =
    eval(parse(x))::AbstractDrillingCost

ArgParse.parse_item(::Type{AbstractExtensionCost}, x::AbstractString) =
    eval(parse(x))::AbstractExtensionCost

ArgParse.parse_item(::Type{DrillingRevenue}, x::AbstractString) =
    eval(parse(x))::DrillingRevenue

ArgParse.parse_item(::Type{Vector{T}}, x::AbstractString) where {T} =
    eval(parse(x))::Vector{T}

ArgParse.parse_item(::Type{T}, x::AbstractString) where {T<:Real} =
    eval(parse(x))::T

function print_parsed_args(x::Dict)
    dset = ["dataset",]
    ctrl = [
        "Mcnstr",
        "Mfull",
        "doCnstr",
        "noFull",
        "noPar",
        "maxtimeCnstr",
        "maxtimeFull",
        "computeStarting",
    ]
    parm = [
        "computeStarting",
        "theta",
    ]
    payoff = [
        "revenue",
        "cost",
        "extension",
        "anticipateT1EV",
        "discount",
    ]
    approx = [
        "numP",
        "numPsi",
        "extendPriceGrid",
        "minTransProb",
    ]

    kys = vcat(dset, ctrl, parm, payoff, approx)
    padlen = maximum(length.(kys))
    padk(k) = rpad(k,padlen)

    println("--------------------------------------")
    println("ARGUMENTS PASSED")
    println("--------------------------------------")
    println("DATASET")
    map(k -> println("  $(padk(k))  =>  $(x[k])"), dset)
    println("ESTIMATION CONTROL")
    map(k -> println("  $(padk(k))  =>  $(x[k])"), ctrl)
    println("PAYOFF")
    map(k -> println("  $(padk(k))  =>  $(x[k])"), payoff)
    println("APPROXIMATION")
    map(k -> println("  $(padk(k))  =>  $(x[k])"), approx)
    println("--------------------------------------")
    print("\n\n")
    return nothing
end
