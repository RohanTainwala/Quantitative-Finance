# Equity Financing & Delta-One

Models the core instruments and economics of a synthetic equity financing desk:
Total Return Swaps, Single Stock Futures, and securities lending. Built to reflect
the risk drivers of a real synthetic sourcing/delta-one book — the desk's P&L
comes from the spread between what it charges clients for synthetic exposure and
its own true cost of funding/borrowing, not from directional market views.

## Files

- **`total_return_swap.py`** — TRS cash flow simulation and P&L attribution
  (financing leg vs. equity return leg vs. dividends), from the desk's perspective.
- **`single_stock_future.py`** — cost-of-carry fair value, implied financing
  rate extraction, and cash-and-carry / reverse cash-and-carry basis trading.
- **`securities_lending.py`** — General Collateral vs. "special" borrow rate
  modeling, and a term-vs-overnight financing decision framework.
- **`notebook.ipynb`** — walks through all three with visuals: a simulated TRS
  P&L path, a cost-of-carry curve, and the rebate-rate-vs-utilization curve
  that defines when a stock "goes special."

## What's simplified vs. a real desk

This is built to demonstrate the core economics correctly, not to replicate
full production infrastructure. Real desks additionally handle:
- Collateral/margin mechanics and counterparty credit risk (CVA/FVA)
- Balance sheet and capital costs (RWA optimization) alongside pure financing spread
- Multi-name basket TRS and index-level netting
- Real-time borrow rate feeds rather than a stylized utilization curve

## Running it

From this folder: `python total_return_swap.py`, `python single_stock_future.py`,
or `python securities_lending.py` to see each run standalone, or open
`notebook.ipynb` in Jupyter for the full walkthrough with charts.
