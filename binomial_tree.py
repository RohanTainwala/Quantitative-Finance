"""
derivatives-pricing/binomial_tree.py
-------------------------------------
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Supports both European and American exercise, which is the main reason
to reach for a tree over closed-form Black-Scholes: American options
have no general closed-form solution because of early-exercise optionality.
"""

from __future__ import annotations
import numpy as np


def binomial_price(
    s: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 500,
    option_type: str = "call",
    exercise: str = "european",
    q: float = 0.0,
) -> float:
    """
    CRR binomial tree price.

    exercise : "european" or "american"
    n_steps  : number of tree steps (accuracy improves ~O(1/n), can oscillate;
               use n_steps in the hundreds for good convergence)
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0 < p < 1):
        raise ValueError(
            "Risk-neutral probability out of (0,1) range — check inputs "
            "(often caused by n_steps too small for given sigma/T)."
        )

    # terminal asset prices
    j = np.arange(n_steps + 1)
    s_terminal = s * (u ** (n_steps - j)) * (d ** j)

    if option_type == "call":
        values = np.maximum(s_terminal - k, 0.0)
    elif option_type == "put":
        values = np.maximum(k - s_terminal, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # backward induction
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

        if exercise == "american":
            j = np.arange(step + 1)
            s_node = s * (u ** (step - j)) * (d ** j)
            if option_type == "call":
                intrinsic = np.maximum(s_node - k, 0.0)
            else:
                intrinsic = np.maximum(k - s_node, 0.0)
            values = np.maximum(values, intrinsic)
        elif exercise != "european":
            raise ValueError("exercise must be 'european' or 'american'")

    return float(values[0])


if __name__ == "__main__":
    from black_scholes import bs_price

    s, k, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

    euro_tree = binomial_price(s, k, r, sigma, T, n_steps=500, option_type="call", exercise="european")
    euro_bs = bs_price(s, k, r, sigma, T, "call")
    print(f"European call — binomial: {euro_tree:.4f}  |  Black-Scholes: {euro_bs:.4f}")

    amer_put = binomial_price(s, k, r, sigma, T, n_steps=500, option_type="put", exercise="american")
    euro_put = binomial_price(s, k, r, sigma, T, n_steps=500, option_type="put", exercise="european")
    print(f"American put: {amer_put:.4f}  |  European put: {euro_put:.4f}  "
          f"(American >= European due to early-exercise value: {amer_put >= euro_put})")
