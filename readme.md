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

$$
\max_w \quad \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
$$

Subject to:

$$
\sum_i w_i = 1, \quad w_i \ge 0 \; \forall i
$$

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

From daily returns $r_t$, the mean daily return is annualized as:

$$
\mu_{annual} \approx (1 + \mu_{daily})^{252} - 1
$$

This mirrors how quantitative models convert historical data into expected returns.

---

### 2️⃣ Building the Annualized Covariance Matrix

The daily covariance matrix $\Sigma_{daily}$ is annualized as:

$$
\Sigma_{annual} \approx \Sigma_{daily} \times 252
$$

This risk model is foundational to:
- Mean–variance optimization
- Risk parity portfolios
- Efficient frontier construction
- Factor and risk-budgeting frameworks

---

### 3️⃣ Portfolio Statistics

Given:
- $w$ = weight vector  
- $\mu$ = vector of annual expected returns  
- $\Sigma$ = annualized covariance matrix  
- $r_f$ = risk-free rate  

Portfolio return:

$$
\mu_p = w^\top \mu
$$

Portfolio variance and volatility:

$$
\sigma_p^2 = w^\top \Sigma w
$$

$$
\sigma_p = \sqrt{w^\top \Sigma w}
$$

Sharpe ratio:

$$
\text{Sharpe}(w) = \frac{\mu_p - r_f}{\sigma_p}
$$

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

---

## 🛠 How to Run the Project (Beginner → Advanced)

This project supports **three ways to run**, depending on your comfort level:

- **Level 1 (Beginner):** run locally with Python + a virtual environment  
- **Level 2 (Recommended):** install once as a CLI command using `pipx`  
- **Level 3 (Most portable):** run in Docker (no Python required)

---

## 🟢 Level 1 — Run Locally (Beginner-Friendly)

### Step 0: Prerequisites
- Python 3.9+ installed
- `pip` available

### Step 1: Clone the repository
Run these commands in your terminal:

    git clone https://github.com/Myan17/Sharpe_optimizer.git
    cd Sharpe_optimizer

### Step 2: Create and activate a virtual environment (recommended)
macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Windows (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

You should now see `(.venv)` in your terminal.

### Step 3: Install dependencies

    pip install -r requirements.txt

### Step 4: Run the optimizer

    python3 sharpe_optimizer.py

Follow the prompts to enter tickers, dates, and the risk-free rate.

---

## 🟡 Level 2 — Install as a Global CLI Tool (Recommended)

This installs the optimizer as a **system command**, so you can run it from anywhere.

### Step 1: Install pipx (one-time)
macOS:

    brew install pipx
    pipx ensurepath

Restart your terminal after `pipx ensurepath`.

### Step 2: Install the Sharpe Optimizer CLI from GitHub

    pipx install git+https://github.com/Myan17/Sharpe_optimizer.git

### Step 3: Run the CLI

    sharpe-opt

### Step 4: Headless mode (no GUI window)
If you are on a server or want to save the plot instead of opening a window:

    HEADLESS=1 sharpe-opt

This will save the weights bar chart as `weights.png`.

Upgrade / uninstall:

    pipx upgrade sharpe-optimizer
    pipx uninstall sharpe-optimizer

---

## 🔴 Level 3 — Run with Docker (No Python Required)

Docker runs the app in an isolated environment, so it behaves the same on any machine.

### Step 0: Prerequisite
- Install Docker Desktop and make sure Docker is running

### Step 1: Build the Docker image (from the repo root)

    docker build -t sharpe-optimizer .

### Step 2: Run the optimizer (interactive prompts)

    docker run -it sharpe-optimizer

### Step 3: Headless mode + export the plot
Run with headless plotting enabled:

    docker run --name so -it -e HEADLESS=1 sharpe-optimizer

Copy the generated plot to your computer:

    docker cp so:/app/weights.png .
    docker rm so

You will now have `weights.png` in your current directory.


---

## 📁 Project Structure 

```text
sharpe_optimizer/
├── sharpe_optimizer/        # Python package
│   ├── __init__.py          # Package initializer
│   └── cli.py               # CLI entry point (sharpe-opt command)
│
├── pyproject.toml           # Package metadata & dependencies
├── requirements.txt         # Dependency list (for local installs)
├── Dockerfile               # Docker image definition (Level 4)
├── .dockerignore            # Excludes unnecessary files from Docker builds
├── .gitignore               # Ignores virtualenvs, caches, artifacts
├── README.md                # Project documentation
└── .venv/                   # Local virtual environment (not committed)
```

---

## 🚀 Why This Project Matters (Quant Career)

By completing this project, you demonstrate:
- How to go from raw price data to optimal portfolio weights
- A practical understanding of portfolio optimization mechanics
- Why unconstrained Sharpe maximization often leads to concentration
- How expected returns and covariance jointly shape risk-return trade-offs

This is a strong foundational project for quantitative finance, asset management, and financial engineering roles.
