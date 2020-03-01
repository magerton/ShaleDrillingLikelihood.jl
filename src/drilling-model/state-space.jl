export LeasedProblem,
    LeasedProblemContsDrill,
    PerpetualProblem

struct LeasedProblem <: AbstractUnitProblem
    dmax::Int   # max number of wells per quarter
    Dmax::Int   # max number of total wells
    τ0max::Int  # last lease regime before final expiration (extension if available)
    τ1max::Int  # initial lease regime if extension
    ext::Int    # length of extenion allowed
end

struct LeasedProblemContsDrill <: AbstractUnitProblem
    dmax::Int
    Dmax::Int
    τ0max::Int
    τ1max::Int
    ext::Int
end

struct PerpetualProblem <: AbstractUnitProblem
    dmax::Int
    Dmax::Int
end

PerpetualProblem(dmax,Dmax,τ0max,τ1max,ext) = PerpetualProblem(dmax,Dmax)

function ==(x::AUP, y::AUP) where {AUP<:Union{LeasedProblem, LeasedProblemContsDrill}}
    x.dmax == y.dmax &&
    x.Dmax == y.Dmax &&
    x.τ0max == y.τ0max &&
    x.τ1max == y.τ1max &&
    x.ext == y.ext
end

==(x::AbstractUnitProblem, y::AbstractUnitProblem) = false

function ==(x::PerpetualProblem, y::PerpetualProblem)
    x.dmax == y.dmax &&
    x.Dmax == y.Dmax
end

# see https://stackoverflow.com/questions/40160120/generic-constructors-for-subtypes-of-an-abstract-type
(::Type{T})(wp::AbstractUnitProblem) where {T<:AbstractUnitProblem} = T(_dmax(wp), _Dmax(wp), _τ0max(wp), _τ1max(wp), _ext(wp))
function (::Type{T})(wp::PerpetualProblem) where {T<:Union{LeasedProblem,LeasedProblemContsDrill}}
    @warn("PerpetualProblem does not have τ0max, τ1max, or ext information!!")
    T(_dmax(wp), _Dmax(wp), _τ0max(wp), _τ1max(wp), _ext(wp))
end

convert(::Type{T}, wp::AbstractUnitProblem) where {T<:AbstractUnitProblem} = T(wp)


_nstates_per_D(wp::AbstractUnitProblem) = 1
_nstates_per_D(wp::LeasedProblemContsDrill) = 2

_dmax(  wp::AbstractUnitProblem) = wp.dmax
_Dmax(  wp::AbstractUnitProblem) = wp.Dmax

_τ0max( wp::AbstractUnitProblem) = wp.τ0max
_τ1max( wp::AbstractUnitProblem) = wp.τ1max
_ext(   wp::AbstractUnitProblem) = wp.ext

_τ0max( wp::PerpetualProblem) = -1
_τ1max( wp::PerpetualProblem) = -1
_ext(   wp::PerpetualProblem) = 0

@inline num_choices(wp, i...) = _dmax(wp, i...)+1

# ----------------------------------
# endpoints of each set of states
#   end_ex1 - Expiration of first lease term
#   end_ex0 - End of last lease term (expiration if present, or regular lease if not)
#   end_lrn - In learning stage, we have drilled _dmax(wp) wells
#   end_inf - Terminal state -- we've drilled everything
#   strt_ex - State where extension starts
# ----------------------------------
@inline end_ex1(wp::LeasedProblemContsDrill) = _τ1max(wp)+1
@inline end_ex0(wp::LeasedProblemContsDrill) = end_ex1(wp) + max(_τ0max(wp),_ext(wp))+1
@inline end_lrn(wp::LeasedProblemContsDrill) = end_ex0(wp) + _dmax(wp)+1
@inline end_inf(wp::LeasedProblemContsDrill) = end_lrn(wp) + 2*_Dmax(wp)
@inline strt_ex(wp::LeasedProblemContsDrill) = end_ex0(wp) - (_ext(wp)-1)     # t+d+2(D)+1 = (t+1) + (d+1) + 2*(D-1) + 1

@inline end_ex1(wp::LeasedProblem) = _τ1max(wp)+1
@inline end_ex0(wp::LeasedProblem) = end_ex1(wp) + max(_τ0max(wp),_ext(wp))+1
@inline end_lrn(wp::LeasedProblem) = end_ex0(wp) + _dmax(wp)+1
@inline end_inf(wp::LeasedProblem) = end_lrn(wp) + _Dmax(wp)+1
@inline strt_ex(wp::LeasedProblem) = end_ex0(wp) - (_ext(wp)-1)     # t+d+2(D)+1 = (t+1) + (d+1) + 2*(D-1) + 1

@inline end_ex1(wp::PerpetualProblem) = 0
@inline end_ex0(wp::PerpetualProblem) = 1
@inline end_lrn(wp::PerpetualProblem) = end_ex0(wp) + _dmax(wp)
@inline end_inf(wp::PerpetualProblem) = end_lrn(wp) + _Dmax(wp) + 1
@inline strt_ex(wp::PerpetualProblem) = 1

# ----------------------------------
# length of states
# ----------------------------------

length(wp::AbstractUnitProblem)  = end_inf(wp)
_nS(    wp::AbstractUnitProblem) = length(wp)
_nSexp( wp::AbstractUnitProblem) = end_lrn(wp)

# ----------------------------------
# regions of state space where we have learning
# ----------------------------------

exploratory_terminal(wp::AbstractUnitProblem) = end_ex0(wp)+1
exploratory_learning(wp::AbstractUnitProblem) = (end_ex0(wp)+1) : end_lrn(wp)

exploratory_terminal(wp::PerpetualProblem) = end_ex0(wp)
exploratory_learning(wp::PerpetualProblem) = end_ex0(wp) : end_lrn(wp)

# ----------------------------------
# indices of where we are in things
# ----------------------------------

@inline ind_ex1(wp::AbstractUnitProblem) = end_ex1(wp)   : -1 : 1
@inline ind_ex0(wp::AbstractUnitProblem) = end_ex0(wp)   : -1 : end_ex1(wp)+1
@inline ind_exp(wp::AbstractUnitProblem) = end_ex0(wp)   : -1 : 1
@inline ind_lrn(wp::AbstractUnitProblem) = end_lrn(wp)   : -1 : end_ex0(wp)+1
@inline ind_inf(wp::AbstractUnitProblem) = end_inf(wp)-1 : -1 : end_lrn(wp)+1

@inline inf_fm_lrn(wp::AbstractUnitProblem) = (end_lrn(wp)+1) .+ (0:_dmax(wp))
@inline function inf_fm_lrn(wp::LeasedProblemContsDrill)
    a = end_lrn(wp)
    it = (a+1, a .+ 2 .* (1:_dmax(wp)))
    return collect(flatten(it))
end

# -------------------------------------------------------------------

"""
    state_if_never_drilled(wp::AbstractUnitProblem, state0::Integer, t::Integer)

At what state will a unit be in after `t` ADDITIONAL periods have passed if it has
decision structure `wp` and a starting `state0`?
"""
function state_if_never_drilled(wp::AbstractUnitProblem, state0::Integer, t::Integer)

    if state0 <= end_ex1(wp)
        if state0 + t <= end_ex1(wp)
            return state0 + t
        elseif state0 + t <= end_ex1(wp) + _ext(wp)
            return (state0 + t) - (end_ex1(wp) - strt_ex(wp)) - 1
        else
            return end_ex0(wp)+1
        end

    elseif state0 <= end_ex0(wp)
        if state0 + t <= end_ex0(wp)
            return state0 + t
        else
            return end_ex0(wp)+1
        end

    else
        return end_ex0(wp)+1
    end
end

function state_if_never_drilled(wp::PerpetualProblem, state0::Integer, t::Integer)
    return 1
end

# ----------------------------------
# which state are we in?
# ----------------------------------

function state_idx(wp::LeasedProblemContsDrill, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    t1 >= 0 && t0 ∉ (0,_ext(wp)) && throw(error("Cannot be in primary + extension simultaneously"))
    t1 >= 0               && return end_ex1(wp) - t1
    t0 >= 0               && return end_ex0(wp) - t0
    D==0  && t0==-1       && return end_ex0(wp) + (1 + 0)
    D<_Dmax(wp) && t0==-1 && return end_lrn(wp) +  2*D + (1-d1)
    D==_Dmax(wp)          && return end_lrn(wp) +  2*D   # drop last +1 since we have d1=1 at terminal
    throw(error("invalid state"))
end

function state_idx(wp::LeasedProblem, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    t1 >= 0 && t0 ∉ (0,_ext(wp)) && throw(error("Cannot be in primary + extension simultaneously"))
    t1 >= 0                && return end_ex1(wp) - t1
    t0 >= 0                && return end_ex0(wp) - t0
    D==0   && t0==-1       && return end_ex0(wp) + 1 + 0
    D<=_Dmax(wp) && t0==-1 && return end_lrn(wp) + 1 + D
    throw(error("invalid state"))
end

function state_idx(wp::PerpetualProblem, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    D==0         && return end_ex0(wp)
    D<=_Dmax(wp) && return end_lrn(wp) + D+1
    throw(error("invalid state"))
end

function state_idx(wp, args::Number... )
    t1,t0,D,d1 = map(Int, args)
    state_idx(wp, t1, t0, D, d1)
end

# ----------------------------------
# information about state
# ----------------------------------

function _regime(wp::AbstractUnitProblem, s::Integer)::Symbol
    s <= 0 && throw(DomainError(s, "s <= 0"))
    s <= end_ex1(wp) && return :primary_WITH_extension
    s <= end_ex0(wp) && return :primary_or_extension
    s == end_ex0(wp) + 1 && return :expired
    s <= end_lrn(wp) && return :learn
    s == end_lrn(wp) + 1 && return :infill_but_0
    s <= end_inf(wp) && return :infill
    throw(DomainError(s))
end

function _horizon(wp::PerpetualProblem, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)  && return :Infinite
    sidx <= end_lrn(wp)  && return :Learning
    sidx <  end_inf(wp)  && return :Infinite
    sidx == end_inf(wp)  && return :Terminal
    throw(DomainError(sidx))
end

function _horizon(wp::LeasedProblem, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)   && return :Finite
    sidx == end_ex0(wp)+1 && return :Terminal
    sidx <= end_lrn(wp)   && return :Learning
    sidx <  end_inf(wp)   && return :Infinite
    sidx == end_inf(wp)   && return :Terminal
    throw(DomainError(sidx))
end

function _horizon(wp::LeasedProblemContsDrill, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)   && return :Finite
    sidx == end_ex0(wp)+1 && return :Terminal
    sidx <= end_lrn(wp)   && return :Learning
    sidx <  end_inf(wp)   && return iseven(end_inf(wp)-sidx) ? :Infinite : :Finite
    sidx == end_inf(wp)   && return :Terminal
    throw(DomainError(sidx))
end

function s_of_D(wp::PerpetualProblem, D::Integer)::UnitRange{Int}
    D ∉ 0:_Dmax(wp) && throw(DomainError(D))
    D == 0          && return 1:end_ex0(wp)
    D <= _Dmax(wp)  && return (end_lrn(wp)+1) .+ (D:D)
end

function s_of_D(wp::LeasedProblem, D::Integer)::UnitRange{Int}
    D ∉ 0:_Dmax(wp) && throw(DomainError(D))
    D == 0          && return 1:end_ex0(wp)+1
    D <= _Dmax(wp)  && return (end_lrn(wp)+1) .+ (D:D)
end

function s_of_D(wp::LeasedProblemContsDrill, D::Integer)::UnitRange{Int}
    D ∉ 0:_Dmax(wp) && throw(DomainError(D))
    D == 0          && return 1:end_ex0(wp)+1
    D < _Dmax(wp)   && return (end_lrn(wp)+2*D) .+ (0:1)
    D == _Dmax(wp)  && return end_inf(wp):end_inf(wp)
end


function _D(wp::LeasedProblemContsDrill, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return  sidx - end_ex0(wp) - 1
    sidx <= end_inf(wp) && return (sidx - end_lrn(wp) - isodd(sidx-end_lrn(wp)) ) / 2
    throw(DomainError(sidx))
end

function _D(wp::LeasedProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return sidx - end_ex0(wp) - 1
    sidx <= end_inf(wp) && return sidx - end_lrn(wp) - 1
    throw(DomainError(sidx))
end

function _D(wp::PerpetualProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return sidx - end_ex0(wp)
    sidx <= end_inf(wp) && return sidx - end_lrn(wp) - 1
    throw(DomainError(sidx))
end


_d1(wp::AbstractUnitProblem, sidx::Integer) = 0

function _d1(wp::LeasedProblemContsDrill, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)+1 && return 0
    sidx <= end_lrn(wp)   && return 1
    sidx <  end_inf(wp)   && return iseven(sidx-end_lrn(wp))
    sidx == end_inf(wp)   && return 1
end


function _τ1(wp::AbstractUnitProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return end_ex1(wp)-sidx
    return -1
end

function _τ0(wp::AbstractUnitProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return _ext(wp)
    sidx <= end_ex0(wp) && return end_ex0(wp)-sidx
    return -1
end

_τ1(wp::PerpetualProblem, sidx::Integer) = -1
_τ0(wp::PerpetualProblem, sidx::Integer) = -1


"Deterministic state  (`τ1`, `τ0` `D`, `d1`)"
struct state
    τ1::Int  # Time remaining in initial term
    τ0::Int  # Time remaining in final term
    D::Int   # Wells drilled to date
    d1::Int  # Drilling last period == 1, no drilling last period == 0
end

state_idx(wp::AbstractUnitProblem, s::state) = state_idx(wp, s.τ1, s.τ0, s.D, s.d1)

# ------------- methods for state ----------

# Pretty print: https://docs.julialang.org/en/latest/manual/types.html#Custom-pretty-printing-1
string(s::state)::String = string((s.τ1, s.τ0, s.D, s.d1))
show(io::IO, s::state) = print(io, string(s))

function ==(s1::state, s2::state)
    s1.τ1 == s2.τ1    &&     s1.τ0 == s2.τ0   &&     s1.D == s2.D    &&    s1.d1 == s2.d1
end

state(wp::AbstractUnitProblem, s::Integer) = state(_τ1(wp,s), _τ0(wp,s), _D(wp,s), _d1(wp,s))

_d1(s::state) = s.d1
_τ1(s::state) = s.τ1
_τ0(s::state) = s.τ0
_D(s::state) = s.D
_sgnext(s::state) = s.τ1 == 0 && s.τ0 > 0
_τrem(s::state) = max(s.τ1,0) + max(s.τ0,0)
_τ11(s::state, wp::AbstractUnitProblem) = 2*_τrem(s)/maxlease(wp)-1
_Dgt0(wp::AbstractUnitProblem, sidx::Integer) = sidx > end_ex0(wp)+1
_Dgt0(s::state) = s.D > 0

stateinfo(wp::AbstractUnitProblem, st::state) = (_d1(st), _Dgt0(st), _sgnext(st), _τ11(st, wp),)

@inline _sgnext(wp::AbstractUnitProblem, sidx::Integer) = sidx == end_ex1(wp)
@inline _sgnext(wp::AbstractUnitProblem, sidx::Integer, d::Integer) = _sgnext(wp,sidx) && d == 0

@deprecate _sign_lease_extension(sidx::Integer,wp::AbstractUnitProblem) _sgnext(wp,sidx)

@inline expires_today(wp::AbstractUnitProblem, sidx::Integer)             = sidx == end_ex0(wp)
@inline expires_today(wp::AbstractUnitProblem, sidx::Integer, d::Integer) = expires_today(wp,sidx) && d==0
@inline expires_today(wp::PerpetualProblem, ::Integer, ::Integer) = false
@inline expires_today(wp::PerpetualProblem, ::Integer) = false

function _τrem(wp::AbstractUnitProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return end_ex1(wp)-sidx + _ext(wp)+1
    sidx <= end_ex0(wp) && return end_ex0(wp)-sidx
    return -1
end

_τrem(wp::PerpetualProblem,sidx::Integer) = -1

max_ext(wp::AbstractUnitProblem) = end_ex0(wp)-strt_ex(wp)+1

maxlease(wp::AbstractUnitProblem) = max( end_ex1(wp)+max_ext(wp), end_ex0(wp)-end_ex1(wp) ) - 1


_τ11(wp::AbstractUnitProblem, sidx::Integer) = 2*_τrem(wp,sidx)/maxlease(wp)-1

state_idx(wp::AbstractStateSpace, s::state) = state_idx(wp, s.τ1, s.τ0, s.D, s.d1)

# ----------------------------------
# actions
# ----------------------------------

function _dmax(wp::AbstractUnitProblem, sidx::Integer)
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return _dmax(wp)
    sidx <= end_lrn(wp) && return 0
    sidx <  end_inf(wp) && return min((_Dmax(wp)-_D(wp,sidx)),_dmax(wp))
    sidx == end_inf(wp) && return 0
    throw(DomainError(sidx))
end

@inline actionspace(wp::AbstractUnitProblem) = 0:_dmax(wp)
@inline actionspace(wp::AbstractUnitProblem, sidx::Integer) = 0:_dmax(wp,sidx)
@inline dp1space(   wp::AbstractUnitProblem, sidx::Integer) = actionspace(wp,sidx) .+ 1
@inline num_choices(wp::AbstractUnitProblem, args... ) = _dmax(wp, args...) + 1

# ----------------------------------
# states
# ----------------------------------

function state_space_vector(wp::LeasedProblemContsDrill)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(error("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,1) for D in 1:_dmax(wp)]  # τmax+1 + (0:dmax) (note overlap!)
    # Infill drilling with immediately prior drilling
    infill0 = [state(-1,-1,0,0)]
    infill  = [state(-1,-1,D,d1) for D in 1:_Dmax(wp)-1 for d1 in 1:-1:0] # τmax + 2 + dmax + (1:2*Dmax-2)
    # Infill terminal
    inf_term = [state(-1,-1,_Dmax(wp),1)] # τmax + 2 + dmax + 2*Dmax - 2 + 1  = τmax + dmax + 2*Dmax + 1
    return [exp1..., exp0..., inf_int..., infill0..., infill..., inf_term...]
end

function state_space_vector(wp::LeasedProblem)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(error("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,0) for D in 1:_dmax(wp)]  # τmax+1 + (0:dmax) (note overlap!)
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,0) for D in 0:_Dmax(wp)] # τmax + 2 + dmax + (1:2*Dmax-2)
    return [exp1..., exp0..., inf_int..., infill...]
end

function state_space_vector(wp::PerpetualProblem)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(error("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,0) for D in 1:_dmax(wp)]
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,0) for D in 0:_Dmax(wp)]
    return [exp1..., exp0..., inf_int..., infill...]
end

# ----------------------------------
# sprime
# ----------------------------------

function sprime(wp::LeasedProblemContsDrill, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex1(wp)
        d == 0 && return strt_ex(wp)
        d >  0 && return end_ex0(wp) + d + 1
    elseif s <= end_ex0(wp)
        d == 0 && return s+1
        d >  0 && return end_ex0(wp) + d + 1
    elseif s == end_ex0(wp)+1
        return s
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + 2*(s-end_ex0(wp)-1)  # was return endpts[3] + 2*d - 1. note d=(s-endpts[2]-1)
    elseif s < end_inf(wp)
        d == 0  && return s + iseven(s-end_lrn(wp))
        d >  0  && return s + 2*d - isodd(s-end_lrn(wp))
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError(s))
    end
end

function sprime(wp::LeasedProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex1(wp)
        d == 0 && return strt_ex(wp)
        d >  0 && return end_ex0(wp) + d + 1
    elseif s <= end_ex0(wp)
        d == 0 && return s+1
        d >  0 && return end_ex0(wp) + d + 1
    elseif s == end_ex0(wp)+1
        return s
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp))
    elseif s < end_inf(wp)
        return s + d
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError(s))
    end
end

function sprime(wp::PerpetualProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex0(wp)
        d == 0 && return s
        d >  0 && return end_ex0(wp) + d
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp)+1)
    elseif s < end_inf(wp)
        return s + d
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError(s))
    end
end

sprimes(wp::AbstractUnitProblem, sidx::Integer) = (sprime(wp,sidx,d) for d in actionspace(wp,sidx))

function post_learning(wp::LeasedProblem, s::Integer, d::Integer)
    if end_ex0(wp) < s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp))
    else
        throw(DomainError(s))
    end
end

function post_learning(wp::PerpetualProblem, s::Integer, d::Integer)
    if end_ex0(wp) < s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp)+1)
    else
        throw(DomainError(s))
    end
end

function post_learning(wp::LeasedProblemContsDrill, s::Integer, d::Integer)
    if end_ex0(wp) < s <= end_lrn(wp)
        return end_lrn(wp) + 2*(s-end_ex0(wp)-1)
    else
        throw(DomainError(s))
    end
end



"retrieve next state, skipping through learning if necessary."
function ssprime(wp::AbstractUnitProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s <= end_ex0(wp)
        return d == 0 ? sprime(wp,s,d) : post_learning(wp, sprime(wp,s,d), d)  # because of LEARNING transition
    elseif s <= end_inf(wp)
        return sprime(wp,s,d)
    else
        throw(DomainError(s, "s > end_inf(wp)"))
    end
end
