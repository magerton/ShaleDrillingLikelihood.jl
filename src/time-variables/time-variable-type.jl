# `ExogTimeVars` holds a regular time series and can be indexed
# with either a a `Date` or an `Integer`

export Quarter, ExogTimeVars, _timevars

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
function time_idx(tv::ExogTimeVars, t::Date)
    ts = _timestamp(tv)
    t in ts || throw(DomainError(t))
    return searchsortedfirst(ts,t)
end
getindex(tv::ExogTimeVars, t::Date) =
    getindex(_timevars(tv), time_idx(tv,t))

date(tv::ExogTimeVars, t::Integer)     = getindex(_timestamp(tv),t)
length(tv::ExogTimeVars) = length(_timestamp(tv))
size(tv::ExogTimeVars{N}) where {N} = length(tv), N
view(tv::ExogTimeVars, idx) = view(_timevars(tv), idx)

import Base: minimum, maximum

minimum(tv::ExogTimeVars{1}) = minimum(_timevars(tv))
maximum(tv::ExogTimeVars{1}) = maximum(_timevars(tv))

function DateQuarter(y::Integer, q::Integer)
    1 <= q <= 4 || throw(DomainError(q))
    return Date(y, 3*(q-1)+1)
end

function DateQuarter(yq::Real)
    y0, q0 = divrem(yq,1)
    y = Int(y0)
    q = Int(q0*4)+1
    return DateQuarter(y,q)
end

function Quarter(i::Integer)
    1 <= i <= 4 || throw(DomainError(i))
    return Month(3*i)
end
