"""
equity-financing/securities_lending.py
------------------------------------------
Stock borrow/lending rate modeling: General Collateral (GC) vs "special"
(hard-to-borrow) rates, and overnight vs. term financing structures.

Directly maps to the JD language: "Pricing new and existing business
including overnight, term transactions" and "Managing and growing
external lender relationships."

Key concepts:
  - GC (General Collateral): easy-to-borrow stocks, rate close to the
    general funding rate (e.g. SOFR - small spread, since the borrower is
    effectively giving the lender cash collateral earning a rebate).
  - Special: hard-to-borrow stocks (high short interest, low float,
    corporate action driven) trade at a much lower (sometimes negative)
    rebate rate -- the harder it is to borrow, the more the borrower pays.
  - Term vs overnight: a term loan locks in a rate for a fixed period,
    protecting the borrower from a stock "going special" mid-trade at the
    cost of typically paying a small premium for that certainty.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BorrowQuote:
    ticker: str
    fed_funds_or_sofr: float   # the general funding/reference rate
    utilization: float          # 0.0-1.0, fraction of available lendable shares out on loan
    days_to_cover: float        # short interest / avg daily volume -- a "squeeze risk" proxy


def rebate_rate(quote: BorrowQuote) -> float:
    """
    Approximate the cash rebate rate a stock lender pays the cash-collateral
    borrower, as a function of utilization.

    Intuition: at low utilization (easy to borrow / GC), the rebate sits
    close to the funding rate minus a small standard fee. As utilization
    climbs toward 100% (hard to borrow), the rebate collapses -- and can
    go negative, meaning the BORROWER pays a fee on top of giving up their
    collateral interest, rather than receiving a rebate.

    This uses a simple convex approximation, not a real quoted curve --
    real desks calibrate this to actual stock loan market data.
    """
    gc_spread = 0.0025  # 25bps standard GC fee, subtracted from reference rate
    base_rebate = quote.fed_funds_or_sofr - gc_spread

    # Convexity: rebate falls off a cliff as utilization -> 1.0
    squeeze_pressure = quote.utilization ** 4  # only bites hard near full utilization
    days_to_cover_pressure = min(quote.days_to_cover / 10, 1.0)  # normalize; >10 days = max pressure

    borrow_fee = squeeze_pressure * days_to_cover_pressure * 0.15  # up to 15% annualized fee for extreme squeezes
    return base_rebate - borrow_fee


def classify_borrow(quote: BorrowQuote) -> str:
    rebate = rebate_rate(quote)
    if rebate > quote.fed_funds_or_sofr - 0.005:
        return "General Collateral (GC) — easy to borrow"
    elif rebate > 0:
        return "Warm — moderately hard to borrow"
    else:
        return "Special / Hard-to-Borrow — negative rebate, borrower pays a fee"


def term_vs_overnight_decision(
    quote: BorrowQuote,
    term_days: int,
    expected_utilization_path: np.ndarray | None = None,
) -> dict:
    """
    Compare locking in a term borrow rate today vs. rolling overnight at
    (potentially worsening) rates, if utilization is expected to rise over
    the term (e.g. ahead of a known catalyst like an index rebalance or
    earnings-driven short interest build).

    expected_utilization_path : optional array of length term_days giving
                                 a forecast utilization path if rolling
                                 overnight; if None, assumes flat utilization.
    """
    term_quote = quote  # rate locked in today
    term_rebate = rebate_rate(term_quote)
    # A term loan usually costs slightly more than the current spot overnight
    # rate, as compensation to the lender for committing supply for the period.
    term_premium = 0.0010  # 10bps annualized premium for the term commitment
    effective_term_rebate = term_rebate - term_premium

    if expected_utilization_path is None:
        expected_utilization_path = np.full(term_days, quote.utilization)

    overnight_rebates = [
        rebate_rate(BorrowQuote(quote.ticker, quote.fed_funds_or_sofr, u, quote.days_to_cover))
        for u in expected_utilization_path
    ]
    avg_overnight_rebate = float(np.mean(overnight_rebates))

    recommendation = (
        "LOCK IN TERM" if effective_term_rebate > avg_overnight_rebate
        else "ROLL OVERNIGHT"
    )

    return {
        "effective_term_rebate": effective_term_rebate,
        "avg_expected_overnight_rebate": avg_overnight_rebate,
        "advantage_bps": (effective_term_rebate - avg_overnight_rebate) * 10_000,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    sofr = 0.045

    easy = BorrowQuote("AAPL", sofr, utilization=0.15, days_to_cover=1.2)
    warm = BorrowQuote("GME", sofr, utilization=0.80, days_to_cover=6.0)
    special = BorrowQuote("SQUEEZE", sofr, utilization=0.98, days_to_cover=12.0)

    for q in [easy, warm, special]:
        r = rebate_rate(q)
        print(f"{q.ticker:<10} utilization={q.utilization:.0%}  days_to_cover={q.days_to_cover:<5}"
              f"  rebate={r:>8.4%}  ->  {classify_borrow(q)}")

    print("\n--- Term vs overnight decision: expecting utilization to rise (earnings in 2 weeks) ---")
    rising_utilization = np.linspace(0.5, 0.95, 14)  # utilization climbing into an earnings date
    decision = term_vs_overnight_decision(warm, term_days=14, expected_utilization_path=rising_utilization)
    for k, v in decision.items():
        print(f"  {k:<32}{v}")
