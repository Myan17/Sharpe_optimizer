#!/usr/bin/env python3
"""
Sharpe Ratio Optimizer

Given a set of assets and a historical date range, this script:

- Downloads adjusted close prices using yfinance
- Computes daily returns
- Estimates:
    * Expected returns (mean daily return, annualized)
    * Covariance matrix of returns (annualized)
- Solves for the long-only portfolio that maximizes Sharpe ratio:
    * max_w  (μ_p - r_f) / σ_p
    * subject to:
        - sum(w_i) = 1
        - w_i >= 0  for all i

This is a classic mean–variance optimization problem used in portfolio construction.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =========================
# Data loading & returns
# =========================

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close (or close) prices for the given tickers and date range.

    We explicitly set auto_adjust=False to preserve the 'Adj Close' field, but
    we gracefully fall back to 'Close' if needed.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError("No data downloaded. Check tickers and date range.")

    # MultiIndex columns: ('Adj Close', 'AAPL'), etc.
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise ValueError(f"Could not find 'Adj Close' or 'Close' in columns: {data.columns}")
    else:
        # Single-level columns
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            raise ValueError(f"Could not find 'Adj Close' or 'Close' in columns: {data.columns}")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily percentage returns from adjusted close prices.

    r_t = (P_t / P_{t-1}) - 1
    """
    returns = prices.pct_change().dropna(how="all")
    return returns


# =========================
# Portfolio statistics
# =========================

def annualize_returns(mean_daily: float, trading_days: int = 252) -> float:
    """
    Convert an average daily return (scalar) to annualized expected return.

    Approximation:
        μ_annual ≈ (1 + μ_daily)^{trading_days} - 1
    """
    return (1.0 + mean_daily) ** trading_days - 1.0


def annualize_covariance(daily_cov: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Annualize a daily covariance matrix.

    Variance scales approximately linearly with time, so:
        Σ_annual ≈ Σ_daily * trading_days
    """
    return daily_cov * trading_days


def portfolio_stats(
    weights: np.ndarray,
    exp_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (must sum to 1).
    exp_returns : np.ndarray
        Expected annual returns for each asset.
    cov_matrix : np.ndarray
        Annualized covariance matrix.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    (ret, vol, sharpe) : tuple of floats
        ret    : expected portfolio return (annualized)
        vol    : portfolio volatility (annualized)
        sharpe : Sharpe ratio = (ret - risk_free_rate) / vol
    """
    # Expected portfolio return: μ_p = w^T μ
    port_ret = float(weights @ exp_returns)

    # Portfolio variance: w^T Σ w
    port_var = float(weights @ cov_matrix @ weights)
    port_vol = np.sqrt(port_var)

    if port_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (port_ret - risk_free_rate) / port_vol

    return port_ret, port_vol, sharpe


# =========================
# Optimization
# =========================

def negative_sharpe(
    weights: np.ndarray,
    exp_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Objective function: negative Sharpe ratio.

    We minimize this using scipy.optimize.minimize with constraints:

        - sum(weights) = 1
        - weights >= 0 (long-only portfolio)
    """
    _, _, sharpe = portfolio_stats(weights, exp_returns, cov_matrix, risk_free_rate)
    return -sharpe  # because we want to maximize Sharpe


def optimize_sharpe(
    exp_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """
    Solve for the long-only maximum Sharpe portfolio.

    Returns:
        optimal_weights : np.ndarray
    """
    num_assets = len(exp_returns)

    # Initial guess: equal weight portfolio
    x0 = np.ones(num_assets) / num_assets

    # Constraint: sum(weights) = 1
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    })

    # Bounds: 0 <= w_i <= 1 (no short selling)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    result = minimize(
        fun=negative_sharpe,
        x0=x0,
        args=(exp_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


# =========================
# Plotting
# =========================

def plot_weights(tickers: List[str], weights: np.ndarray) -> None:
    """
    Plot a bar chart of portfolio weights.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tickers, weights)
    ax.set_title("Optimal Portfolio Weights (Max Sharpe, Long-Only)")
    ax.set_ylabel("Weight")
    plt.tight_layout()
    plt.show()


# =========================
# CLI main
# =========================

def main() -> None:
    """
    CLI entrypoint.

    Steps:
    1. Ask user for tickers, date range, risk-free rate.
    2. Download prices and compute daily returns.
    3. Estimate expected annual returns and annualized covariance.
    4. Optimize weights to maximize Sharpe ratio.
    5. Print portfolio stats and weights.
    6. Plot weight allocation.
    """
    print("=== Sharpe Ratio Optimizer (Long-Only) ===")

    tickers_input = input("Enter tickers (space-separated, e.g. META NVDA SPY): ").strip()
    tickers = tickers_input.split()

    if len(tickers) < 2:
        print("ERROR: Please provide at least 2 tickers.")
        sys.exit(1)

    start = input("Enter START date (YYYY-MM-DD): ").strip()
    end = input("Enter END date   (YYYY-MM-DD): ").strip()

    rf_str = input("Enter annual risk-free rate in %, e.g. 4 for 4%% [0]: ").strip()
    try:
        rf_rate = float(rf_str) / 100.0 if rf_str else 0.0
    except ValueError:
        print("WARN: Invalid risk-free rate, using 0.0%")
        rf_rate = 0.0

    # --- Download prices and compute returns ---
    prices = download_prices(tickers, start, end)
    returns = compute_daily_returns(prices)

    # --- Estimate expected returns & covariance (annualized) ---
    mean_daily_returns = returns.mean()
    exp_returns = mean_daily_returns.apply(annualize_returns).values  # vector μ

    cov_daily = returns.cov()
    cov_annual = annualize_covariance(cov_daily).values               # matrix Σ

    print("\n--- Estimated Annualized Expected Returns ---")
    for t, mu in zip(tickers, exp_returns):
        print(f"{t}: {mu:.2%}")

    print("\n--- Annualized Covariance Matrix ---")
    print(pd.DataFrame(cov_annual, index=tickers, columns=tickers).round(6))

    # --- Optimize Sharpe ---
    optimal_weights = optimize_sharpe(exp_returns, cov_annual, risk_free_rate=rf_rate)

    port_ret, port_vol, port_sharpe = portfolio_stats(
        optimal_weights, exp_returns, cov_annual, risk_free_rate=rf_rate
    )

    print("\n=== Optimal Long-Only Max Sharpe Portfolio ===")
    for t, w in zip(tickers, optimal_weights):
        print(f"{t}: {w:.2%}")

    print("\n--- Portfolio Stats (Annualized) ---")
    print(f"Expected Return: {port_ret:.2%}")
    print(f"Volatility     : {port_vol:.2%}")
    print(f"Sharpe Ratio   : {port_sharpe:.3f} (rf = {rf_rate:.2%})")

    # --- Plot weights ---
    plot_weights(tickers, optimal_weights)


if __name__ == "__main__":
    main()
