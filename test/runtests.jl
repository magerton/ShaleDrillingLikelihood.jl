using Revise
using Base.Threads

println("\n==============================")
println("using $(nthreads()) threads")

include("threadutils.jl")
# include("sum-functions.jl")
#
# include("data/data-structure.jl")
# include("data/drilling.jl")
# include("data/overall.jl")

# include("drilling-model/flow.jl")
# include("likelihood/royalty.jl")
# include("likelihood/production.jl")
# include("likelihood/drilling.jl")
# include("likelihood/overall.jl")
