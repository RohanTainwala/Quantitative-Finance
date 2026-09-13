"""
derivatives-pricing/monte_carlo_pricer.py
-------------------------------------------
Monte Carlo pricer for European options, built on the shared GBM engine
in core/simulate.py. Reports a standard error alongside the price, since
unlike Black-Scholes or the tree, MC prices are statistical estimates.

Also includes finite-difference Greeks (bump-and-reprice), which is how
Greeks are typically computed for products that don't have closed-form
sensitivities (exotics, path-dependent payoffs).
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from simulate import gbm_paths  # noqa: E402


def mc_price(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Monte Carlo price for a European option under risk-neutral GBM.
    n_steps=1 is sufficient for European payoffs (only S_T matters);
    kept as a parameter so this function can be reused for path-dependent
    payoffs later by increasing n_steps and changing the payoff function.

    Returns
    -------
    (price, standard_error)
    """
    paths = gbm_paths(s0=s, mu=r, sigma=sigma, T=T, n_steps=n_steps,
                       n_paths=n_paths, antithetic=True, seed=seed)
    s_T = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(s_T - k, 0.0)
    elif option_type == "put":
        payoff = np.maximum(k - s_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, se


def mc_greeks_bump(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 200_000,
    bump_s: float = 0.01,
    bump_sigma: float = 0.001,
    seed: int = 42,
) -> dict:
    """
    Finite-difference (bump-and-reprice) Greeks using COMMON RANDOM NUMBERS
    (same seed across bumps) to cancel Monte Carlo noise in the difference —
    without this, the finite-difference Greeks would be far too noisy to use.
    """
    h_s = s * bump_s

    p_up, _ = mc_price(s + h_s, k, r, sigma, T, option_type, n_paths, seed=seed)
    p_down, _ = mc_price(s - h_s, k, r, sigma, T, option_type, n_paths, seed=seed)
    p_mid, _ = mc_price(s, k, r, sigma, T, option_type, n_paths, seed=seed)

    delta = (p_up - p_down) / (2 * h_s)
    gamma = (p_up - 2 * p_mid + p_down) / (h_s**2)

    p_vega_up, _ = mc_price(s, k, r, sigma + bump_sigma, T, option_type, n_paths, seed=seed)
    vega = (p_vega_up - p_mid) / bump_sigma

    return {"delta": delta, "gamma": gamma, "vega": vega, "price": p_mid}


if __name__ == "__main__":
    from black_scholes import bs_price, bs_greeks

    s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

    price, se = mc_price(s, k, r, sigma, T, "call", n_paths=500_000, seed=42)
    bs = bs_price(s, k, r, sigma, T, "call")
    print(f"MC call price: {price:.4f} +/- {1.96*se:.4f} (95% CI)  |  Black-Scholes: {bs:.4f}")

    mc_g = mc_greeks_bump(s, k, r, sigma, T, "call", n_paths=200_000)
    bs_g = bs_greeks(s, k, r, sigma, T, "call")
    print(f"MC delta: {mc_g['delta']:.4f}  |  BS delta: {bs_g['delta']:.4f}")
    print(f"MC gamma: {mc_g['gamma']:.4f}  |  BS gamma: {bs_g['gamma']:.4f}")
    print(f"MC vega:  {mc_g['vega']:.4f}  |  BS vega:  {bs_g['vega']:.4f}")
