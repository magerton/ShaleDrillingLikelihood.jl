# -----------------------------------------
# update [I - TV] operator based on current prices
# -----------------------------------------

function update_IminusTVp!(tmpv::DCDPTmpVars{T,M}, ddm::DynamicDrillModel{S,PF,M}, q0::AbstractVector) where {T,S,M<:Matrix,PF}
    IminusTVp = IminusTEVp(tmpv)
    ztrans = ztransition(ddm)
    β = discount(ddm)

    n = checksquare(IminusTVp)
    n == length(q0) == checksquare(ztrans) || throw(DimensionMismatch())

    @inbounds for j in OneTo(n)
        @simd for i in OneTo(n)
            IminusTVp[i,j] = (i==j) - ztrans[i,j] * β * q0[j]
        end
    end
end


function update_IminusTVp!(tmpv::DCDPTmpVars{T,M}, ddm::DynamicDrillModel{S,PF,M}, q0::AbstractVector) where {T,S,M<:SparseMatrixCSC,PF}
    IminusTVp = IminusTEVp(tmpv)
    ztrans = ztransition(ddm)
    β = discount(ddm)

    n = checksquare(IminusTVp)
    n == length(q0) == checksquare(ztrans) || throw(DimensionMismatch())

    ztrans_rows = rowvals(ztrans)
    ztrans_vals = nonzeros(ztrans)
    IminusTVp_rows = rowvals( IminusTVp)
    IminusTVp_vals = nonzeros(IminusTVp)

    # consider eliminmating this check?
    length(ztrans_rows) == length(IminusTVp_rows) || throw(DimensionMismatch(
        "length(ztrans_rows) = $(length(ztrans_rows)) but length(IminusTVp_rows) = $(length(IminusTVp_rows))"
    ))

    # IminusTVp_rows == ztrans_rows || throw(error("IminusTVp rows not the same as ztrans"))

    fill!(IminusTVp, 0)
    @inbounds for j in OneTo(n)
        @simd for nzi in nzrange(IminusTVp, j)
            i = IminusTVp_rows[nzi]
            IminusTVp_vals[nzi] = (i==j) - ztrans_vals[nzi] * β * q0[j]
            # x = -ztrans_vals[nzi] * β * q0[j]
            # IminusTVp_vals[nzi] = i==j  ?  1+x  :  x
        end
    end

end
