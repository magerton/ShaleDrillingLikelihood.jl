# some constants
# ----------------------------------------------------------------

# From Gulen et al (2015) "Production scenarios for the Haynesville..."
const GATH_COMP_TRTMT_PER_MCF   = 0.42 + 0.07
const MARGINAL_TAX_RATE = 0.402

# other calculations
const REAL_DISCOUNT_AND_DECLINE = 0x1.8e06b611ed4d8p-1  #  0.777394952476835= sum( β^((t+5-1)/12) q(t)/Q(240) for t = 1:240)

const STARTING_α_ψ      = 0x1.7587cc6793516p-2 # 0.365
const STARTING_log_ogip = 0x1.401755c339009p-1 # 0.625
const STARTING_α_t      = 0.01948

const TIME_TREND_BASE = 2008

# -----------------------------------------
# some functions
# -----------------------------------------

# chebshev polynomials
# See http://www.aip.de/groups/soe/local/numres/bookcpdf/c5-8.pdf
@inline checkinterval(x::Real,min::Real,max::Real) =  min <= x <= max || throw(DomainError("x = $x must be in [$min,$max]"))
@inline checkinterval(x::Real) = checkinterval(x,-1,1)

@inline cheb0(z::Real) = one(z)
@inline cheb1(z::Real) = clamp(z,-1,1)
@inline cheb2(z::Real) = (x = cheb1(z); return 2*x^2 - 1)
@inline cheb3(z::Real) = (x = cheb1(z); return 4*x^3 - 3*x)
@inline cheb4(z::Real) = (x = cheb1(z); return 8*(x^4 - x^2) + 1)
