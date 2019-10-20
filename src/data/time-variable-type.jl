struct ExogTimeVars{N, ZTup<:NTuple{N,Real}, OR<:OrdinalRange}
    timevars::Vector{ZTup}
    timestamp::OR
    function ExogTimeVars(timevars::Vector{ZTup}, timestamp::OR) where {N, ZTup<:NTuple{N,Real}, OR<:OrdinalRange}
        length(timevars) .== length(timestamp) || throw(DimensionMismatch())
        new{N,ZTup,OR}(timevars,timestamp)
    end
end


_timevars(tv::ExogTimeVars) = tv.timevars
_timestamp(tv::ExogTimeVars) = tv.timestamp

getindex(tv::ExogTimeVars, t) = getindex(_timevars(tv), t)
function getindex(tv::ExogTimeVars, t::Date)
    ts = _timestamp(tv)
    t in ts || throw(DomainError(t))
    return getindex(_timevars(tv), searchsortedfirst(ts,t))
end

date(tv::ExogTimeVars, t::Integer)     = getindex(_timestamp(tv),t)
length(tv::ExogTimeVars) = length(_timestamp(tv))
size(tv::ExogTimeVars{N}) where {N} = length(tv), N
view(tv::ExogTimeVars, idx) = view(_timevars(tv), idx)


function DateQuarter(y::Integer, q::Integer)
    1 <= q <= 4 || throw(DomainError())
    return Date(y, 3*(q-1)+1)
end


function Quarter(i::Integer)
    1 <= i <= 4 || throw(DomainError())
    return Month(3*i)
end
