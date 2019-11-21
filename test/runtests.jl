using Revise
using Base.Threads

println("using $(nthreads()) threads")

module CompileShaleDrillingLikelihood
    using ShaleDrillingLikelihood
end

include("sum-functions.jl")
include("threadutils.jl")

include("data/data-structure.jl")
include("data/drilling.jl")
include("data/overall.jl")

include("drilling-model/state-space.jl")

include("drilling-model/test-flow.jl")
include("drilling-model/drilling-flow.jl")

include("likelihood/drilling.jl")
include("likelihood/overall.jl")
