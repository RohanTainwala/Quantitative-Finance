"""
equity-financing/single_stock_future.py
------------------------------------------
Single Stock Futures (SSF): fair value via cost-of-carry, implied financing
rate extraction, and basis trading (cash-and-carry arbitrage) mechanics.

An SSF is a delta-one product: an exchange-listed future on a single stock.
Its fair value is driven entirely by the cost of carrying the underlying
stock to the future's expiry -- financing cost minus dividends received.

Fair value:  F = S * exp((r - q) * T)
  S : spot price
  r : financing/repo rate (cost to borrow cash to buy the stock)
  q : dividend yield (income received from holding the stock)
  T : time to expiry (years)

If the future trades away from fair value, a "cash-and-carry" or "reverse
cash-and-carry" arbitrage becomes available -- exactly the kind of relative
value opportunity a delta-one/synthetic financing desk looks for.
"""

from __future__ import annotations
import numpy as np


def ssf_fair_value(s: float, r: float, q: float, T: float) -> float:
    """Cost-of-carry fair value of a single stock future."""
    return s * np.exp((r - q) * T)


def implied_financing_rate(f_market: float, s: float, q: float, T: float) -> float:
    """
    Back out the financing/repo rate the market is implying, given an
    observed future price. This is what a trader watches to see whether
    the market is pricing financing richer or cheaper than the desk's
    own funding cost.

    F = S * exp((r - q)T)  =>  r = ln(F/S)/T + q
    """
    return np.log(f_market / s) / T + q


def basis_trade_pnl(
    s: float,
    f_market: float,
    r_desk: float,
    q: float,
    T: float,
    notional_shares: float = 10_000,
) -> dict:
    """
    Evaluate a cash-and-carry (or reverse cash-and-carry) basis trade:
    buy the cheaper side, sell the richer side, hold to expiry.

    r_desk : the desk's OWN true financing/repo rate (its real cost of funds)
             -- compared against what the market future implies.

    If the future is trading RICH (implied rate > desk's true rate):
      -> desk SELLS the future, BUYS the stock (cash-and-carry), earns the spread
    If the future is trading CHEAP (implied rate < desk's true rate):
      -> desk BUYS the future, SHORTS the stock (reverse cash-and-carry)
    """
    fair_value = ssf_fair_value(s, r_desk, q, T)
    implied_rate = implied_financing_rate(f_market, s, q, T)
    rate_richness_bps = (implied_rate - r_desk) * 10_000  # in basis points

    mispricing = f_market - fair_value
    direction = "SELL future / BUY stock (cash-and-carry)" if mispricing > 0 else \
                "BUY future / SHORT stock (reverse cash-and-carry)" if mispricing < 0 else \
                "No arbitrage — future is fair"

    # Locked-in arbitrage profit per share if held to expiry (ignoring transaction costs/margin)
    arb_profit_per_share = abs(mispricing)
    total_arb_profit = arb_profit_per_share * notional_shares

    return {
        "fair_value": fair_value,
        "market_price": f_market,
        "mispricing_per_share": mispricing,
        "implied_financing_rate": implied_rate,
        "desk_true_financing_rate": r_desk,
        "richness_bps": rate_richness_bps,
        "trade_direction": direction,
        "arb_profit_per_share": arb_profit_per_share,
        "total_arb_profit": total_arb_profit,
    }


if __name__ == "__main__":
    s = 100.0
    r_desk = 0.045   # desk's true funding cost, e.g. SOFR + small spread
    q = 0.02         # 2% annual dividend yield
    T = 90 / 365      # 3-month future

    fv = ssf_fair_value(s, r_desk, q, T)
    print(f"Spot: {s}  |  Fair value (3m SSF): {fv:.4f}")
    print(f"Implied cost of carry: {(fv - s):.4f}  ({(fv/s - 1)*100:.3f}% over 3 months)\n")

    # Scenario: the future is quoted RICH relative to the desk's funding cost
    f_market_rich = 101.30
    result = basis_trade_pnl(s, f_market_rich, r_desk, q, T, notional_shares=50_000)
    print("--- Scenario: future trading rich ---")
    for k, v in result.items():
        print(f"  {k:<28}{v}")

    # Scenario: the future is quoted CHEAP
    f_market_cheap = 100.10
    print("\n--- Scenario: future trading cheap ---")
    result2 = basis_trade_pnl(s, f_market_cheap, r_desk, q, T, notional_shares=50_000)
    for k, v in result2.items():
        print(f"  {k:<28}{v}")
