"""
derivatives-pricing/black_scholes.py
-------------------------------------
Closed-form Black-Scholes-Merton pricing and analytical Greeks for
European calls and puts.

Reference: Black & Scholes (1973), Merton (1973) dividend extension.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def _d1_d2(s: float, k: float, r: float, q: float, sigma: float, T: float):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton price.

    s : spot price
    k : strike
    r : risk-free rate (continuously compounded)
    sigma : volatility (annualized)
    T : time to expiry in years
    option_type : "call" or "put"
    q : continuous dividend yield
    """
    if T <= 0:
        intrinsic = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
        return intrinsic

    d1, d2 = _d1_d2(s, k, r, q, sigma, T)
    disc_s = s * np.exp(-q * T)
    disc_k = k * np.exp(-r * T)

    if option_type == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    elif option_type == "put":
        return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict:
    """
    Analytical Greeks: Delta, Gamma, Vega, Theta, Rho.

    Conventions:
      - Vega  is per 1.00 (100%) change in vol; divide by 100 for "per 1 vol point"
      - Theta is per year; divide by 365 for "per calendar day"
      - Rho   is per 1.00 (100%) change in rate; divide by 100 for "per 1% rate move"
    """
    d1, d2 = _d1_d2(s, k, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = disc_q * pdf_d1 / (s * sigma * np.sqrt(T))
    vega = s * disc_q * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * k * disc_r * norm.cdf(d2)
            + q * s * disc_q * norm.cdf(d1)
        )
        rho = k * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * k * disc_r * norm.cdf(-d2)
            - q * s * disc_q * norm.cdf(-d1)
        )
        rho = -k * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(
    price: float,
    s: float,
    k: float,
    r: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Back out implied volatility from a market price using Newton-Raphson
    (falls back to bisection if Newton fails to converge).
    """
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        model_price = bs_price(s, k, r, sigma, T, option_type, q)
        vega = bs_greeks(s, k, r, sigma, T, option_type, q)["vega"]
        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)

    # Newton failed / didn't converge -> bisection fallback
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        model_price = bs_price(s, k, r, mid, T, option_type, q)
        if abs(model_price - price) < tol:
            return mid
        if model_price > price:
            hi = mid
        else:
            lo = mid
    return mid


if __name__ == "__main__":
    s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    price = bs_price(s, k, r, sigma, T, "call")
    greeks = bs_greeks(s, k, r, sigma, T, "call")
    print(f"BS call price: {price:.4f}")
    print("Greeks:", {g: round(v, 4) for g, v in greeks.items()})

    iv = implied_vol(price, s, k, r, T, "call")
    print(f"Recovered implied vol: {iv:.4f} (should be ~0.2000)")

    
