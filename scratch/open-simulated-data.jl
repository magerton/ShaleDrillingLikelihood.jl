using SparseArrays
using JLD2
using ShaleDrillingLikelihood

modulepath = dirname(pathof(ShaleDrillingLikelihood))
filepath = joinpath(modulepath, "..", "scratch", "simulated-data.jld2")

file = jldopen(filepath, "r")
    in_j1ptr   = file["j1ptr"]
    in_j2ptr   = file["j2ptr"]
    in_tptr    = file["tptr"]
    in_jtstart = file["jtstart"]
    in_j1chars = file["j1chars"]
    in_ichars  = file["ichars"]
    in_y       = file["y"]
    in_x       = file["x"]
    in_zchars  = file["zchars"]
    in_zrng    = file["zrng"]
    in_psis    = file["psis"]
    in_ztrans  = file["ztrans"]
close(file)
