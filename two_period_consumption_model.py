# streamlit_app.py
# Two-Period Consumption Model — Interactive Streamlit App
# --------------------------------------------------------
# This app lets you explore optimal consumption in a 2-period model with CRRA utility.
# You can vary income (y1, y2), interest rate r, time preference beta, risk aversion sigma,
# and the time gap between incomes (tau). All calculations are closed-form and consistent
# with the Euler equation and the intertemporal budget constraint.

import math
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Core Model Functions
# -----------------------------
def present_lifetime_value(y1: float, y2: float, r: float, tau: float = 1.0) -> float:
    """
    Present value of lifetime resources when the second income arrives tau periods from now.
    r is the per-period net real interest rate.
    """
    return y1 + y2 / ((1.0 + r) ** tau)


def euler_ratio(beta: float, r: float, sigma: float, tau: float = 1.0) -> float:
    """
    Returns k such that c2 = k * c1 from the Euler equation:
      u'(c1) = beta * (1+r)^tau * u'(c2)
    For CRRA (sigma != 1): c2/c1 = [beta * (1+r)^tau]^(1/sigma)
    For log utility (sigma == 1): c2/c1 = beta * (1+r)^tau
    """
    if abs(sigma - 1.0) < 1e-12:
        return beta * ((1.0 + r) ** tau)
    return (beta * ((1.0 + r) ** tau)) ** (1.0 / sigma)


def solve_c1(y1: float, y2: float, r: float, sigma: float, beta: float, tau: float = 1.0) -> float:
    """
    Solve for optimal c1 using:
      PV = c1 + c2 / (1+r)^tau,   c2 = k * c1
      => c1 = PV / [1 + k / (1+r)^tau]
    where k is given by euler_ratio.
    """
    pv = present_lifetime_value(y1, y2, r, tau)
    k = euler_ratio(beta, r, sigma, tau)
    denom = 1.0 + k / ((1.0 + r) ** tau)
    return pv / denom


def euler_consumption(c1: float, beta: float, r: float, sigma: float, tau: float = 1.0) -> float:
    """Compute c2 from the Euler condition."""
    return c1 * euler_ratio(beta, r, sigma, tau)


def utility(c1: float, c2: float, beta: float, sigma: float) -> float:
    """
    Lifetime utility with CRRA (or log when sigma == 1):
      U = u(c1) + beta * u(c2),
      u(c) = c^(1-sigma)/(1-sigma)  if sigma != 1
      u(c) = ln(c)                  if sigma == 1
    """
    if c1 <= 0 or c2 <= 0:
        return float("-inf")
    if abs(sigma - 1.0) < 1e-12:
        return math.log(c1) + beta * math.log(c2)
    return (c1 ** (1.0 - sigma) + beta * (c2 ** (1.0 - sigma))) / (1.0 - sigma)


def simulate_two_period(y1: float, y2: float, r: float, beta: float, sigma: float, tau: float = 1.0):
    """Convenience wrapper returning all key outcomes."""
    c1 = solve_c1(y1, y2, r, sigma, beta, tau)
    c2 = euler_consumption(c1, beta, r, sigma, tau)
    pv = present_lifetime_value(y1, y2, r, tau)
    a2 = y1 - c1  # period-1 savings (could be <0 if borrowing)
    u = utility(c1, c2, beta, sigma)
    # Budget check residual (numerical noise near zero is OK)
    budget_resid = (c1 + c2 / ((1.0 + r) ** tau)) - pv
    return {
        "C1": c1,
        "C2": c2,
        "PV": pv,
        "A2": a2,
        "U": u,
        "budget_residual": budget_resid,
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Two-Period Consumption Model", page_icon="📈")

st.title("📈 Two-Period Consumption Model (CRRA Utility)")
st.caption(
    "Interactively explore optimal consumption across two periods. "
    "Adjust incomes, interest rate, time preference, risk aversion, and the time between incomes."
)

with st.expander("Model setup and formulas"):
    st.markdown(
        r"""
**Income evolution:**

- If the "Flat income assumption" box is checked:  
  $$
  y_2 = y_1
  $$
- Otherwise, with wage growth rate $g$:  
  $$
  y_2 = y_1 (1 + g)
  $$

**Budget constraint (present value):**

$$
c_1 + \frac{c_2}{(1+r)^{\tau}} = y_1 + \frac{y_2}{(1+r)^{\tau}}
$$

**Euler equation:**

$$
u'(c_1) = \beta (1+r)^{\tau} \, u'(c_2)
$$

**For CRRA utility with coefficient $\sigma$:**

$$
\frac{c_2}{c_1} =
\begin{cases}
\left[\beta(1+r)^{\tau}\right]^{1/\sigma}, & \text{if } \sigma \neq 1 \\
\beta(1+r)^{\tau}, & \text{if } \sigma = 1
\end{cases}
$$

**Closed-form solution:**

$$
c_1 = \frac{y_1 + \dfrac{y_2}{(1+r)^{\tau}}}{1 + \dfrac{k}{(1+r)^{\tau}}}
$$

$$
c_2 = k \, c_1
$$

$$
k =
\begin{cases}
\left[\beta(1+r)^{\tau}\right]^{1/\sigma}, & \text{if } \sigma \neq 1 \\
\beta(1+r)^{\tau}, & \text{if } \sigma = 1
\end{cases}
$$
        """,
        unsafe_allow_html=False,
    )

st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    y1 = st.number_input("Income today (y₁)", min_value=0.0, value=60.0, step=1.0, format="%.2f")
    flat_income = st.checkbox("Flat income assumption (y₂ = y₁)", value=False)
    if not flat_income:
        g = st.number_input("Wage growth rate g", min_value=-1.0, value=0.05, step=0.01, format="%.2f")
        y2 = y1 * (1 + g)
    else:
        y2 = y1
    r = st.number_input("Real interest rate r (per period)", min_value=0.0, value=0.017, step=0.001, format="%.3f")
with col2:
    beta = st.slider("Time preference β", min_value=0.50, max_value=1.00, value=0.96, step=0.001)
    sigma = st.slider("Risk aversion σ (CRRA)", min_value=0.10, max_value=5.00, value=2.00, step=0.10)
    tau = st.slider("Time between incomes τ (in periods)", min_value=0.25, max_value=5.00, value=1.00, step=0.25,
                    help="If y₂ arrives after τ periods. For annual r, τ counts years; for monthly r, τ counts months, etc.")

st.divider()

results = simulate_two_period(y1, y2, r, beta, sigma, tau)

c1 = results["C1"]
c2 = results["C2"]
pv = results["PV"]
a2 = results["A2"]
u = results["U"]
resid = results["budget_residual"]

st.subheader("Results")

m1, m2, m3 = st.columns(3)
m1.metric("Optimal consumption today (c₁)", f"{c1:,.4f}")
m2.metric("Optimal consumption in future (c₂)", f"{c2:,.4f}")
m3.metric("Present value of resources (PV)", f"{pv:,.4f}")

m4, m5 = st.columns(2)
m4.metric("Savings after period 1 (a₂ = y₁ − c₁)", f"{a2:,.4f}")
m5.metric("Lifetime utility", f"{u:,.6f}")

# Tiny budget check to show numerical consistency
st.caption(f"Budget residual (should be ≈ 0): {resid:.6e}")

# Simple visualization
st.subheader("Visualization")
st.bar_chart(
    {
        "Amount": {
            "y₁ (today’s income)": y1,
            "c₁ (today’s consumption)": c1,
            "a₂ (savings end of period 1)": a2,
            "PV of y₂": y2 / ((1.0 + r) ** tau),
            "c₂/(1+r)^τ (PV of future consumption)": c2 / ((1.0 + r) ** tau),
        }
    }
)

with st.expander("📝 Notes and Tips (Click to expand)"):
    st.markdown(
        """
**⚠️ Important Notes and Tips**

- **Units are arbitrary** (you can treat y₁ and y₂ as currency in any scale).  
- **y₂ is determined by your choice:**  
    - If "Flat income assumption" is checked, then y₂ = y₁.  
    - Otherwise, y₂ = y₁ × (1 + g), where g is the wage growth rate you enter.  
- **r is a *net* real rate per period.** If you have an annual real rate of 1.7% and τ in years, set r = 0.017 and τ accordingly.  
- **τ scales time.** If the second income arrives sooner or later, τ adjusts discounting and the Euler condition accordingly.  
- Ensure **c₁ and c₂ remain positive** for meaningful utility (the app guards by reporting −∞ utility if not).

---
*Click the expander to hide these notes.*
        """,
        unsafe_allow_html=False,
    )

# --- Budget Line Plot ---
st.subheader("Budget Line Plot (C₁ vs C₂)")
fig, ax = plt.subplots()
# Budget line: C1 + C2/(1+r) = Y1 + Y2/(1+r)
c1_vals = np.linspace(0, pv, 200)
c2_vals = (pv - c1_vals) * (1 + r)
ax.plot(c1_vals, c2_vals, label="Budget Line", color="blue")
ax.fill_between(c1_vals, 0, c2_vals, color="blue", alpha=0.1)
ax.plot([c1], [c2], 'ro', label="Optimal (C₁*, C₂*)")
x_annot = pv * 0.6
y_annot = (pv - x_annot) * (1 + r)
ax.annotate("Slope = - (1 + r)", xy=(x_annot, y_annot), xytext=(x_annot, y_annot + pv*0.1),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("C₁ (today's consumption)")
ax.set_ylabel("C₂ (future consumption)")
ax.legend()
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    The blue line shows all combinations of current (C₁) and future (C₂) consumption that exactly exhaust your lifetime resources (the budget constraint).  
    The shaded region is the set of feasible choices.  
    The red dot marks the optimal consumption pair.  
    **Look for:**  
    How the slope (−(1+r)) reflects the tradeoff between consuming today and in the future.
    """
)

# --- Indifference Curves ---
st.subheader("Indifference Curves (CRRA Utility)")
fig, ax = plt.subplots()
ax.plot(c1_vals, c2_vals, label="Budget Line", color="blue")
for delta in [-0.2, 0, 0.2]:
    u_level = u + delta * abs(u)
    c2_curve = lambda c1: ((u_level * (1 - sigma) - c1**(1-sigma)) / beta)**(1/(1-sigma)) if sigma != 1 else np.exp((u_level - np.log(c1))/beta)
    c2_vals_indiff = []
    for c1v in c1_vals:
        try:
            val = c2_curve(c1v)
            c2_vals_indiff.append(val if val > 0 else np.nan)
        except:
            c2_vals_indiff.append(np.nan)
    ax.plot(c1_vals, c2_vals_indiff, '--', label=f"Indiff. curve (U ≈ {u_level:.2f})")
ax.plot([c1], [c2], 'ro', label="Optimal (C₁*, C₂*)")
ax.set_xlabel("C₁")
ax.set_ylabel("C₂")
ax.set_ylim(bottom=0)
ax.legend()
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    The dashed lines are indifference curves—each shows combinations of C₁ and C₂ that yield the same lifetime utility.  
    The blue line is the budget constraint, and the red dot is the optimal point where the highest possible indifference curve touches the budget line.  
    **Look for:**  
    How the shape of indifference curves changes with risk aversion (σ), and how the optimal point is where the budget line is tangent to an indifference curve.
    """
)

# --- Comparative Statics Heatmap ---
st.subheader("Comparative Statics: Utility Heatmap (r, β)")
r_grid = np.linspace(0.0, 0.2, 40)
beta_grid = np.linspace(0.5, 1.0, 40)
U_grid = np.zeros((len(beta_grid), len(r_grid)))
for i, beta_val in enumerate(beta_grid):
    for j, r_val in enumerate(r_grid):
        res = simulate_two_period(y1, y2, r_val, beta_val, sigma, tau)
        U_grid[i, j] = res["U"]
fig, ax = plt.subplots()
sns.heatmap(U_grid, xticklabels=np.round(r_grid, 3), yticklabels=np.round(beta_grid, 2),
            cmap="viridis", ax=ax, cbar_kws={'label': 'Lifetime Utility'})
ax.set_xlabel("Real interest rate r")
ax.set_ylabel("Time preference β")
ax.set_title("Lifetime Utility U")
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    This heatmap shows how lifetime utility varies as you change the real interest rate (r) and time preference (β).  
    Brighter colors indicate higher utility.  
    **Look for:**  
    How utility responds to changes in patience (β) and the interest rate (r), and where utility is maximized.
    """
)

# --- Consumption Smoothing Paths ---
st.subheader("Consumption Smoothing Paths")
scenarios = [
    ("Baseline", y1, y2, r, beta, sigma),
    ("Income shock (↓y₁)", y1*0.7, y2, r, beta, sigma),
    ("Policy shock (↑β)", y1, y2, r, beta+0.03, sigma),
    ("Interest rate shock (↑r)", y1, y2, r+0.03, beta, sigma),
    ("Risk aversion shock (↑σ)", y1, y2, r, beta, sigma+1.0),
]
c1s, c2s, labels = [], [], []
for label, y1_s, y2_s, r_s, beta_s, sigma_s in scenarios:
    res = simulate_two_period(y1_s, y2_s, r_s, beta_s, sigma_s, tau)
    c1s.append(res["C1"])
    c2s.append(res["C2"])
    labels.append(label)
fig, ax = plt.subplots()
ax.bar(np.arange(len(labels))-0.15, c1s, width=0.3, label="C₁")
ax.bar(np.arange(len(labels))+0.15, c2s, width=0.3, label="C₂")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=20)
ax.set_ylabel("Consumption")
ax.legend()
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    This bar plot compares optimal consumption today (C₁) and in the future (C₂) across different economic scenarios.  
    **Look for:**  
    How shocks to income, policy, interest rate, or risk aversion affect the smoothing of consumption between periods.
    """
)

# --- Savings vs Interest Rate Curve ---
st.subheader("Savings vs Interest Rate")
r_range = np.linspace(0, 0.2, 50)
a2s = [simulate_two_period(y1, y2, r_val, beta, sigma, tau)["A2"] for r_val in r_range]
fig, ax = plt.subplots()
ax.plot(r_range, a2s, color="purple")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Real interest rate r")
ax.set_ylabel("Savings A₂ = y₁ − c₁")
ax.set_title("Savings vs Interest Rate")
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    This line plot shows how savings (A₂) change as the real interest rate (r) varies.  
    **Look for:**  
    Where the curve crosses zero (the switch from borrowing to saving), and how higher interest rates incentivize more saving.
    """
)

# --- Utility Component Decomposition ---
st.subheader("Utility Component Decomposition")
if sigma != 1:
    u1 = c1**(1-sigma)/(1-sigma)
    u2 = beta * c2**(1-sigma)/(1-sigma)
else:
    u1 = np.log(c1)
    u2 = beta * np.log(c2)
fig, ax = plt.subplots()
ax.bar(["u₁", "βu₂"], [u1, u2], color=["#1f77b4", "#ff7f0e"])
ax.set_ylabel("Utility Contribution")
ax.set_title("Decomposition of Lifetime Utility")
st.pyplot(fig)
st.markdown(
    """
    **What you see:**  
    This bar plot breaks down total lifetime utility into the contribution from period 1 (u₁) and the discounted contribution from period 2 (βu₂).  
    **Look for:**  
    The relative size of each bar, which shows how much each period's consumption contributes to overall utility.
    """
)

