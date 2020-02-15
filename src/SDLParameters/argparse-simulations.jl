# Overall structure
#----------------------------

export parse_commandline_counterfactuals, print_parsed_args_counterfactuals

function arg_settings_counterfactuals()
    s = ArgParseSettings()

    @add_arg_table! s begin

        "--jld2"
            arg_type = String
            default = "E:/projects/ShaleDrillingResults/2019-12/16481962/estimation-results-16481962-no-rigs-data_all_leases.jld2"

        "--noPar"
            action = :store_true

        "--dateStop"
            arg_type = Float64
            default = 2016.75

        "--techYearZero"
            arg_type = Int
            default = 2007

        "--rFileDir"
            arg_type = String
            default = "./"
    end
    return s
end

parse_commandline_counterfactuals() = parse_args(arg_settings_counterfactuals())

function print_parsed_args_counterfactuals(x::Dict)

    kys = keys(x)
    padlen = maximum(length.(kys))
    padk(k) = rpad(k,padlen)

    println("--------------------------------------")
    println("ARGUMENTS PASSED")
    println("--------------------------------------")
    for (k,v) in x
        println("  $(padk(k))  =>  $v")
    end
    println("--------------------------------------")
    print("\n\n")
    return nothing
end
