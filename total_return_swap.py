"""
equity-financing/total_return_swap.py
----------------------------------------
Total Return Swap (TRS) modeling: valuation, cash flow projection, and
P&L attribution.

A TRS lets the "receiver" get the full economic exposure of an underlying
stock (price return + dividends) without owning it, while paying the
"payer" (typically a bank's equity financing desk) a financing rate on
the notional. This is the core synthetic-sourcing product on a delta-one
desk: the desk owns/borrows the actual stock to hedge, and earns the
spread between what it charges the client and its own funding cost.

Two legs:
  1. Equity leg   : price return of the underlying + dividends received
  2. Financing leg: notional * (reference_rate + spread) * day_count_fraction

From the DESK'S perspective (as payer of the equity leg, receiver of financing):
  Desk P&L = Financing leg received - Equity leg paid out - actual dividend/borrow cost
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TRSContract:
    notional: float           # notional value at trade inception (e.g. $10,000,000)
    initial_price: float      # underlying price at trade inception
    reference_rate: float     # e.g. SOFR, annualized
    spread: float             # financing spread charged over reference rate (annualized, e.g. 0.0150 = 150bps)
    start_date_days: int = 0  # days since epoch, for reset scheduling (kept simple: just day counts)
    reset_freq_days: int = 30 # how often the financing leg resets/settles (e.g. monthly)


@dataclass
class TRSCashflow:
    day: int
    price: float
    price_return_pnl: float          # cumulative price return P&L on the equity leg since last reset
    dividend_pnl: float               # dividends accrued since last reset
    financing_leg: float              # financing amount owed by the equity-return receiver
    net_settlement: float             # net cash flow at this reset (positive = receiver pays desk)


def simulate_trs_cashflows(
    contract: TRSContract,
    price_path: np.ndarray,
    daily_dividend: np.ndarray | None = None,
) -> list[TRSCashflow]:
    """
    Walk a single simulated (or actual historical) daily price path and
    generate the periodic TRS settlement cash flows.

    price_path : array of daily prices, price_path[0] == contract.initial_price
    daily_dividend : optional array of per-share cash dividends paid on
                      each day (0 on non-ex-dividend days)
    """
    n_days = len(price_path) - 1
    if daily_dividend is None:
        daily_dividend = np.zeros(n_days + 1)

    shares_equivalent = contract.notional / contract.initial_price
    cashflows = []

    last_reset_day = 0
    last_reset_price = contract.initial_price
    accrued_dividends = 0.0

    for day in range(1, n_days + 1):
        accrued_dividends += daily_dividend[day] * shares_equivalent

        is_reset_day = (day - last_reset_day) >= contract.reset_freq_days or day == n_days
        if is_reset_day:
            price_now = price_path[day]
            price_return_pnl = (price_now - last_reset_price) * shares_equivalent
            days_elapsed = day - last_reset_day
            financing_leg = contract.notional * (contract.reference_rate + contract.spread) * (days_elapsed / 365)

            # From the desk's perspective: it receives financing_leg, and pays out
            # price appreciation + dividends to the TRS receiver (client).
            # If the stock fell, the client instead PAYS the desk the depreciation.
            net_settlement = financing_leg - price_return_pnl - accrued_dividends

            cashflows.append(TRSCashflow(
                day=day,
                price=price_now,
                price_return_pnl=price_return_pnl,
                dividend_pnl=accrued_dividends,
                financing_leg=financing_leg,
                net_settlement=net_settlement,
            ))

            last_reset_day = day
            last_reset_price = price_now
            accrued_dividends = 0.0

    return cashflows


def desk_pnl_breakdown(cashflows: list[TRSCashflow], actual_borrow_cost: float = 0.0) -> dict:
    """
    Summarize the desk's total economics across all resets.

    actual_borrow_cost : total actual cost the desk paid to borrow/hold the
                          hedge stock over the life of the trade (e.g. stock
                          loan fee) -- this is what the financing spread is
                          meant to cover and profit over.
    """
    total_financing_received = sum(cf.financing_leg for cf in cashflows)
    total_price_pnl_paid = sum(cf.price_return_pnl for cf in cashflows)
    total_dividends_paid = sum(cf.dividend_pnl for cf in cashflows)
    net_settlement = sum(cf.net_settlement for cf in cashflows)

    return {
        "total_financing_received": total_financing_received,
        "total_price_pnl_paid_to_client": total_price_pnl_paid,
        "total_dividends_paid_to_client": total_dividends_paid,
        "net_settlement_to_desk": net_settlement,
        "actual_borrow_cost": actual_borrow_cost,
        "desk_net_pnl": net_settlement - actual_borrow_cost,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
    from simulate import gbm_paths

    # Set up a 1-year TRS on a $10mm notional
    contract = TRSContract(
        notional=10_000_000,
        initial_price=100.0,
        reference_rate=0.045,   # e.g. SOFR
        spread=0.0150,          # 150bps financing spread charged to client
        reset_freq_days=30,     # monthly resets
    )

    # Simulate a realistic 1-year daily price path (single path for this demo)
    paths = gbm_paths(s0=100, mu=0.08, sigma=0.25, T=1.0, n_steps=365, n_paths=1, seed=7)
    price_path = paths[0]

    # Assume 4 quarterly dividends of $0.50/share
    daily_div = np.zeros(len(price_path))
    for q in [90, 180, 270, 360]:
        if q < len(daily_div):
            daily_div[q] = 0.50

    cashflows = simulate_trs_cashflows(contract, price_path, daily_div)

    print(f"{'Day':>5}{'Price':>10}{'PriceRtnPnL':>14}{'DivPnL':>10}{'FinLeg':>12}{'NetSettle':>14}")
    for cf in cashflows:
        print(f"{cf.day:>5}{cf.price:>10.2f}{cf.price_return_pnl:>14,.0f}"
              f"{cf.dividend_pnl:>10,.0f}{cf.financing_leg:>12,.0f}{cf.net_settlement:>14,.0f}")

    breakdown = desk_pnl_breakdown(cashflows, actual_borrow_cost=15_000)
    print("\nDesk P&L breakdown over the life of the trade:")
    for k, v in breakdown.items():
        print(f"  {k:<32}{v:>14,.2f}")
