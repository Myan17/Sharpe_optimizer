# Sharpe Ratio Optimizer (Long Term Only)

This project builds a **maximum Sharpe ratio portfolio** using historical returns and risk estimates from market data.  

---

## 🎯 Project Goal

Given:

- A list of tickers (e.g., `META NVDA SPY`)  
- A historical date range  
- An annual risk-free rate (e.g., 4%)  

The script:

1. Downloads historical prices using **yfinance**
2. Computes **daily returns**
3. Estimates:
   - **Expected annualized returns** for each asset  
   - **Annualized covariance matrix** of returns  
4. Solves for the **long-only portfolio** (no short selling) that **maximizes the Sharpe ratio**
5. Prints:
   - Asset-level expected returns  
   - The covariance matrix  
   - Optimal portfolio weights  
   - Portfolio return, volatility, and Sharpe ratio  
6. Plots a **bar chart of portfolio weights**

The optimization problem is:

\[
\max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\]

Subject to:

\[
\sum_i w_i = 1,\quad w_i \ge 0\ \forall i
\]

This is solved using SciPy’s **SLSQP** optimizer, which supports both equality constraints and bounds.

---

## 🧪 Example Run (Using Your Actual Output)

**Inputs used:**

- Tickers: `META NVDA SPY`  
- Start Date: `2020-01-01`  
- End Date: `2024-01-01`  
- Risk-Free Rate: `4%` (annual)

---

### 📈 Estimated Annualized Expected Returns

    META: 27.45%
    NVDA: 96.52%
    SPY : 14.67%

**Interpretation:**

- **NVDA** had extremely strong realized returns over this period.
- **META** also performed very well.
- **SPY** (broad market) had lower but more typical equity returns.

---

### 🧮 Annualized Covariance Matrix

              META      NVDA       SPY
    META   0.218838  0.143474  0.067149
    NVDA   0.143474  0.294072  0.087966
    SPY    0.067149  0.087966  0.051198

**Interpretation:**

- **Diagonal entries** are variances:
  - NVDA has the highest variance → most volatile.
  - SPY has the lowest → most stable.
- **Off-diagonals** are covariances:
  - All are positive → these assets tend to move together.
  - Diversification exists, but it is not perfect.

---

### 🏆 Optimal Long-Only Max Sharpe Portfolio

    META: 0.00%
    NVDA: 100.00%
    SPY : 0.00%

The optimizer allocates **100% to NVDA** because:

- NVDA’s expected return (~96.5%) is far higher than META and SPY.
- Even with higher volatility, its **Sharpe ratio dominates** any diversified mix of the three, under the constraints:
  - fully invested (sum of weights = 1)  
  - no short-selling (weights ≥ 0)

This is typical behavior when running **unconstrained (or lightly constrained) Sharpe maximization** on a universe where one asset massively outperformed historically.

---

### 📊 Portfolio Performance (Annualized)

    Expected Return: 96.52%
    Volatility     : 54.23%
    Sharpe Ratio   : 1.706 (rf = 4.00%)

- A Sharpe ratio of **1.7+** is extremely strong for a long-only portfolio.
- In practice, funds often cap single-name weights, sectors, or industries to avoid this kind of concentration.  
- Here we deliberately **do not** add those constraints so you can see the “pure” max-Sharpe solution.

## 🧠 What You Learn from This Project

### 1️⃣ Estimating Expected Returns

From daily returns \( r_t \), you estimate mean daily return \( \mu_{\text{daily}} \) per asset and annualize it:

\[
\mu_{\text{annual}} \approx (1 + \mu_{\text{daily}})^{252} - 1
\]

This teaches you how quants turn historical return data into **expected returns** for allocation models.

---

### 2️⃣ Building the Annualized Covariance Matrix

You compute the daily covariance matrix \( \Sigma_{\text{daily}} \) and annualize it:

\[
\Sigma_{\text{annual}} \approx \Sigma_{\text{daily}} \times 252
\]

This is the same **risk model input** used in:

- Mean–variance optimization  
- Risk parity portfolios  
- Efficient frontier modeling  
- Factor and risk-budgeting frameworks  

---

### 3️⃣ Portfolio Statistics

Given:

- \( w \) = weight vector  
- \( \mu \) = vector of annual expected returns  
- \( \Sigma \) = annualized covariance matrix  
- \( r_f \) = risk-free rate  

You compute:

- **Portfolio return**  
  \[
  \mu_p = w^\top \mu
  \]

- **Portfolio variance & volatility**  
  \[
  \sigma_p^2 = w^\top \Sigma w,\quad \sigma_p = \sqrt{w^\top \Sigma w}
  \]

- **Sharpe ratio**  
  \[
  \text{Sharpe}(w) = \frac{\mu_p - r_f}{\sigma_p}
  \]

These are then used inside the optimizer.

---

### 4️⃣ Numerical Optimization with SciPy

You implement a function:

- `negative_sharpe(weights, exp_returns, cov_matrix, risk_free_rate)`

and pass it into:

- `scipy.optimize.minimize` using the **SLSQP** method

With:

- **Constraint:** sum of weights = 1  
- **Bounds:** each weight in [0, 1]

This is exactly how many real-world allocation engines are built.

---

## 🛠 How to Run This Project

From the `sharpe_optimizer` directory:

1. **Install dependencies (in your venv):**

    pip install -r requirements.txt

   Where `requirements.txt` contains:

   - pandas  
   - numpy  
   - matplotlib  
   - yfinance  
   - scipy  

2. **Run the script:**

    python3 sharpe_optimizer.py

3. **Provide inputs when prompted:**

    Enter tickers (space-separated, e.g. META NVDA SPY): META NVDA SPY  
    Enter START date (YYYY-MM-DD): 2020-01-01  
    Enter END date   (YYYY-MM-DD): 2024-01-01  
    Enter annual risk-free rate in %, e.g. 4 for 4% [0]: 4  

You’ll see:

- Estimated annual returns per ticker  
- Annualized covariance matrix  
- Optimal portfolio weights  
- Portfolio stats (return, volatility, Sharpe)  
- A bar plot of the optimal weights  

---

## 📁 Project Structure

    sharpe_optimizer/
    ├── sharpe_optimizer.py   # Main script: loads data, optimizes Sharpe, plots weights
    ├── requirements.txt      # Python dependencies
    └── README.md             # This documentation

---

## 🚀 Why This Project Matters (Quant Career)

By finishing this project, you now understand:

- How to go from **raw price series → portfolio weights**
- How optimization **really works** under the hood
- Why unconstrained Sharpe maximization often leads to **concentrated portfolios**
- How expected return and covariance jointly determine portfolio risk-return trade-offs


