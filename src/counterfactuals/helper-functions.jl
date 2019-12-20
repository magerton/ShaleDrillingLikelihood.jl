export approx_stdnorm

"""
    @declarezero(T, args...)

To create scalar `Float64` variables `x` and `y` that equal 0, run `@declarezero Float64 x y`

taken from <https://stackoverflow.com/questions/37229630/define-multiple-variables-of-same-type-in-one-line-in-julia>

see also <https://stackoverflow.com/questions/31313040/julia-automatically-generate-functions-and-export-them>
"""
macro declarezero(T, args...)
    r = quote end
    for var in args
        # push!(r.args, :( $(esc(var))::$T = zero($T)) )
        push!(r.args, :( $(esc(var)) = zero($T)) )
    end
    r
end



"""
    @addStructFields(T, flds...)

Adds `flds` of type `T` to struct. Note that `T` can have parameters.

# Example

```julia
struct myStruct{T<:Number}
    @addStructFields Vector{T} x y
end
```
"""
macro addStructFields(T, flds::Symbol...)
    r = quote end
    for f in flds
        e = quote $(esc(f))::$(esc(T)) end
        push!(r.args, e)
    end
    return r
end


"declare `vars...` equal to `T`"
macro declareVariables(T, vars::Symbol...)
    r = quote end
    for v in vars
        e = quote $(esc(v)) = $(esc(T)) end
        push!(r.args, e)
    end
    return r
end


"Fill column `j` of `obj.f` with zeros for `f` in `flds"
macro zeroColumn_j(obj, j, flds::Symbol...)
    r = quote end
    for f in flds
        e = quote ShaleDrillingModel.zero!(
            view(
                $(esc(obj)).$(f),
                :, $(esc(j))
            )
        )
        end
        push!(r.args, e)
    end
    return r
end




function approx_stdnorm(z::AbstractVector{<:Real}; L::Integer=11, κ::Real=1e-10)

    n = length(z)

    scale_factor = maximum(abs.(z))
    scaled_moments = [m for m in MarkovTransitionMatrices.NormCentralMoment(L, 1.0/scale_factor)]

    ΔT = Matrix{Float64}(undef,n,L)
    MarkovTransitionMatrices.ΔTmat!(ΔT, z./scale_factor, scaled_moments)

    q = max.(normpdf.(z), κ)
    qp = similar(q)

    for l in L:-2:1
        @views J = discreteApprox!(qp, Vector{Float64}(undef,l), Vector{Float64}(undef,l), Vector{Float64}(undef,n), q, ΔT[:,1:l])
        isfinite(J) && break
    end

    return qp
end
