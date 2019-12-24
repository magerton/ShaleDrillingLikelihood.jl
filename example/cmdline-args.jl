using Revise
using ShaleDrillingLikelihood
using ShaleDrillingLikelihood.SDLParameters

parsed_args = parse_commandline()

println(keys(parsed_args))
print_parsed_args(parsed_args)
