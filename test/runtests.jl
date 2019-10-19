using Revise
using Base.Threads

println("using $(nthreads()) threads")

# include("sum-functions.jl")

# include("data/data-structure.jl")
# include("data/drilling.jl")

# include("drilling-model/flow.jl")
# include("likelihood/royalty.jl")
# include("likelihood/production.jl")
include("likelihood/drilling.jl")
# include("likelihood/overall.jl")
