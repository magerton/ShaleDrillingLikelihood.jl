# using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood: DrillingCost_TimeFE
using JLD2

JLDFILES = [
    # "E:/projects/ShaleDrillingResults/2019-12/16481962/estimation-results-16481962-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16482113/estimation-results-16482113-no-rigs-data_first_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16482515/estimation-results-16482515-no-rigs-data_first_lease_only_all_leased.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16482550/estimation-results-16482550-no-rigs-data_last_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16486124/estimation-results-16486124-WITH-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16486177/estimation-results-16486177-WITH-rigs-data_first_lease_only_all_leased.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16486227/estimation-results-16486227-WITH-rigs-data_first_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16486272/estimation-results-16486272-WITH-rigs-data_last_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16544476/estimation-results-16544476-WITH-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16627219/estimation-results-16627219-no-rigs-data_first_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16627762/estimation-results-16627762-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16653809/estimation-results-16653809-WITH-rigs-data_first_lease_only.jld2",
    # "E:/projects/ShaleDrillingResults/2019-12/16677572/estimation-results-16677572-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17477499/estimation-results-17477499-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2020-01/17489762/estimation-results-17489762-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2020-01/17491570/estimation-results-17491570-no-rigs-data_all_leases.jld2",

    # "E:/projects/ShaleDrillingResults/2020-01/17491625/estimation-results-17491625-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2020-01/17491863/estimation-results-17491863-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2020-01/17491864/estimation-results-17491864-no-rigs-data_all_leases.jld2",
    # "E:/projects/ShaleDrillingResults/2020-01/17493674/estimation-results-17493674-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17493679/estimation-results-17493679-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17525858/estimation-results-17525858-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17529657/estimation-results-17529657-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17608936/estimation-results-17608936-no-rigs-data_all_leases.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17608940/estimation-results-17608940-no-rigs-data_first_lease_only_all_leased.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17608948/estimation-results-17608948-no-rigs-data_first_lease_only.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17608951/estimation-results-17608951-no-rigs-data_last_lease_only.jld2",
    "E:/projects/ShaleDrillingResults/2020-01/17608965/estimation-results-17608965-no-rigs-data_all_leases.jld2",
]

for JLD2FILE in JLDFILES

    println_time_flush("Loading results from $JLD2FILE")
    file = jldopen(JLD2FILE, "r")
        DATAPATH = file["DATAPATH"]
        M        = file["M"]
        ddmnovf  = file["ddm_novf"]
        theta    = file["theta1"]
    close(file)
end
