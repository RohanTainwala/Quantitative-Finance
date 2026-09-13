# Quantitative Finance

A collection of quantitative finance work spanning algorithmic trading strategy
development in MQL5 and derivatives/delta one pricing and risk modeling in Python.
Built to demonstrate both practical trading system development and the underlying
mathematics of derivatives pricing, risk, and equity financing.

## Contents

### 1. MQL5 Trading Strategies

Algorithmic trading strategies developed and backtested in MQL5 for MetaTrader 5,
built during my time as a Forex Analyst at Bacera Co. Pty. Ltd.

- **Martingale.mq5** – Martingale strategy with variations and exponential step up
- **M1_Streak_Fade.mq5** – Scalping strategy that fades short term reversals

Strategies were evaluated using Sharpe ratio, Z score, maximum drawdown, and other
statistical performance measures via MetaTrader 5's Strategy Tester. Full backtest
reports are included as Excel exports (`ReportTester-*.xlsx`).

### 2. Quant Portfolio (Python)

Located in `quant-portfolio/`. Derivative pricing, stochastic simulation, and
equity financing modeling, each built as tested code plus a notebook that
explains the math and shows the results, not just a script that runs.

| Project | What it covers |
|---|---|
| `derivatives-pricing/` | Black Scholes, binomial tree, and Monte Carlo option pricing, cross validated against each other, with analytical and finite difference Greeks. Includes a live pricer that pulls real market data. |
| `equity-financing/` | Total Return Swap cash flow and P&L modeling, Single Stock Future fair value and basis trading, and securities lending rate modeling (General Collateral versus special, term versus overnight financing). |
| `stochastic-processes/` (planned) | Geometric Brownian Motion, Ornstein Uhlenbeck mean reversion, and jump diffusion path simulation and comparison. |
| `volatility-surface/` (planned) | Implied volatility extraction from real option chains and 3D smile and skew visualization. |
| `monte-carlo-risk/` (planned) | Portfolio Value at Risk and Conditional Value at Risk via Monte Carlo, historical, and parametric methods. |

See `quant-portfolio/README.md` for full setup instructions, design notes, and
what is simplified relative to a real trading desk.

## Why both are here

The MQL5 work shows systematic strategy development and backtesting discipline.
The Python work shows the pricing and risk mathematics underneath derivatives
and delta one products. Together they cover both ends of quantitative finance:
building and testing a trading system, and understanding the instruments and
risk that sit underneath it.

## Background

I'm Rohan Tainwala. More of my finance work, including free calculators for
options pricing, Monte Carlo simulation, and DCF modeling, is available at
[finingale.com](https://finingale.com).

## Disclaimer

This repository is for educational and portfolio purposes. Nothing here is
investment advice, and the MQL5 strategies are not intended for live trading
without further validation.
