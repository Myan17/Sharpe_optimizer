# Sharpe Ratio Optimizer (Long-Only)

This project builds a **maximum Sharpe ratio portfolio** using historical market data, estimated returns, and risk metrics.

The optimizer constructs a **fully invested, long-only portfolio** (no short selling) that maximizes risk-adjusted return over a user-specified historical period.

---

## 🎯 Project Goal

Given:
- A list of tickers (e.g., `META NVDA SPY`)
- A historical date range
- An annual risk-free rate (e.g., `4%`)

the script:

- Downloads historical price data using `yfinance`
- Computes daily returns
- Estimates:
  - Expected **annualized returns** for each asset
  - **Annualized covariance matrix** of returns
- Solves for the **long-only portfolio** that maximizes the Sharpe ratio
- Prints:
  - Asset-level expected returns
  - The covariance matrix
  - Optimal portfolio weights
  - Portfolio return, volatility, and Sharpe ratio
- Plots a bar chart of the optimal portfolio weights

---

## 📐 Optimization Formulation

The optimization problem solved is:

\[
\max_w \quad \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\]

Subject to:

\[
\sum_i w_i = 1, \quad w_i \ge 0 \; \forall i
\]

This ensures the portfolio is:
- Fully invested
- Long-only (no short positions)

The problem is solved using **SciPy’s SLSQP optimizer**, which supports both equality constraints and bound constraints.

---

## 🧪 Example Run (Using Actual Output)

### Inputs
- **Tickers:** META NVDA SPY  
- **Start Date:** 2020-01-01  
- **End Date:** 2024-01-01  
- **Risk-Free Rate:** 4% (annual)

---

## 📈 Estimated Annualized Expected Returns

| Ticker | Expected Return |
|------|----------------|
| META | 27.45% |
| NVDA | 96.52% |
| SPY  | 14.67% |

**Interpretation:**
- NVDA exhibited exceptionally strong realized returns during this period.
- META also performed very well.
- SPY (broad market exposure) delivered lower but more typical equity returns.

---

## 🧮 Annualized Covariance Matrix

|        | META | NVDA | SPY |
|------|------|------|-----|
| META | 0.2188 | 0.1435 | 0.0671 |
| NVDA | 0.1435 | 0.2941 | 0.0880 |
| SPY  | 0.0671 | 0.0880 | 0.0512 |

**Interpretation:**
- Diagonal elements represent **variances**:
  - NVDA is the most volatile.
  - SPY is the most stable.
- Off-diagonal elements represent **covariances**:
  - All are positive, indicating assets generally move together.
  - Diversification exists, but it is imperfect.

## 🏆 Optimal Long-Only Max Sharpe Portfolio

| Asset | Weight |
|-----|--------|
| META | 0.00% |
| NVDA | 100.00% |
| SPY  | 0.00% |

**Why does this happen?**

- NVDA’s expected return (~96.5%) is dramatically higher than META and SPY.
- Even with higher volatility, NVDA’s Sharpe ratio dominates any diversified combination under the constraints:
  - Fully invested
  - No short selling

This behavior is **typical** when performing unconstrained (or lightly constrained) Sharpe maximization on assets with extreme historical outperformance.

---

## 📊 Portfolio Performance (Annualized)

- **Expected Return:** 96.52%  
- **Volatility:** 54.23%  
- **Sharpe Ratio:** 1.706 (rf = 4.00%)

A Sharpe ratio above **1.7** is extremely strong for a long-only portfolio.

In real-world portfolio management, practitioners often impose:
- Single-asset weight caps
- Sector or industry limits

These constraints are **intentionally excluded** here to demonstrate the *pure* max-Sharpe solution.

---

## 🧠 Key Concepts Learned

### 1️⃣ Estimating Expected Returns

From daily returns \( r_t \), the mean daily return is annualized as:

\[
\mu_{\text{annual}} \approx (1 + \mu_{\text{daily}})^{252} - 1
\]

This mirrors how quantitative models convert historical data into expected returns.

---

### 2️⃣ Building the Annualized Covariance Matrix

The daily covariance matrix \( \Sigma_{\text{daily}} \) is annualized as:

\[
\Sigma_{\text{annual}} \approx \Sigma_{\text{daily}} \times 252
\]

This risk model is foundational to:
- Mean-variance optimization
- Risk parity portfolios
- Efficient frontier construction
- Factor and risk-budgeting frameworks

---

### 3️⃣ Portfolio Statistics

Given:
- \( w \): weight vector  
- \( \mu \): expected returns vector  
- \( \Sigma \): covariance matrix  
- \( r_f \): risk-free rate  

You compute:

**Portfolio return**
\[
\mu_p = w^\top \mu
\]

**Portfolio variance and volatility**
\[
\sigma_p^2 = w^\top \Sigma w, \quad \sigma_p = \sqrt{w^\top \Sigma w}
\]

**Sharpe ratio**
\[
\text{Sharpe}(w) = \frac{\mu_p - r_f}{\sigma_p}
\]

---

### 4️⃣ Numerical Optimization with SciPy

The optimizer minimizes a custom objective:

```python
negative_sharpe(weights, exp_returns, cov_matrix, risk_free_rate)
```

using:
- `scipy.optimize.minimize`
- Method: **SLSQP**
- Constraint: sum of weights = 1
- Bounds: each weight ∈ [0, 1]

---

## 🛠 How to Run the Project

From the `sharpe_optimizer` directory:

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
yfinance
scipy
```

### 2️⃣ Run the script
```bash
python3 sharpe_optimizer.py
```

### 3️⃣ Provide inputs when prompted
```text
Enter tickers (space-separated, e.g. META NVDA SPY): META NVDA SPY
Enter START date (YYYY-MM-DD): 2020-01-01
Enter END date (YYYY-MM-DD): 2024-01-01
Enter annual risk-free rate in %, e.g. 4 for 4% [0]: 4
```

You will see:
- Estimated annual returns
- Annualized covariance matrix
- Optimal portfolio weights
- Portfolio performance metrics
- A bar chart of portfolio weights

---

## 📁 Project Structure

```text
sharpe_optimizer/
├── sharpe_optimizer.py   # Data loading, optimization, and plotting
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Why This Project Matters (Quant Career)

By completing this project, you demonstrate:
- How to go from raw price data to optimal portfolio weights
- A practical understanding of portfolio optimization mechanics
- Why unconstrained Sharpe maximization often leads to concentration
- How expected returns and covariance jointly shape risk-return trade-offs

This is a strong foundational project for quantitative finance, asset management, and financial engineering roles.
