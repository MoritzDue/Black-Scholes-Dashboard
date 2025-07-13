import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# =====================
# Option Pricing & Greeks
# =====================
def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

# Black-Scholes Greeks calculation function
def calculate_greek_matrix(S, K_vals, T, r, vol_vals, greek_name, premium):
    matrix = []
    for vol in vol_vals:
        row = []
        for K in K_vals:
            d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)

            delta_call = norm.cdf(d1)
            delta_put = delta_call - 1
            gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            theta_call = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            theta_put = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

            greek_map = {
                "Delta": (delta_call, delta_put),
                "Gamma": (gamma, gamma),
                "Vega": (vega, vega),
                "Theta": (theta_call, theta_put),
                "Rho": (rho_call, rho_put),
            }

            # You can decide if you want call or put here, or both separately
            # For heatmaps, let's show call Greek by default
            value = greek_map[greek_name][0]
            row.append(value)
        matrix.append(row)
    return np.array(matrix)

def calculate_greeks(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "Delta": (delta_call, delta_put),
        "Gamma": (gamma, gamma),
        "Vega": (vega, vega),
        "Theta": (theta_call, theta_put),
        "Rho": (rho_call, rho_put),
    }

# =====================
# Alternative Pricing Models
# =====================

def monte_carlo_option_pricing(S, K, T, r, sigma, n_simulations=10000, option_type="call"):
    np.random.seed(42)
    z = np.random.standard_normal(n_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

def binomial_option_pricing(S, K, T, r, sigma, steps=100, option_type="call"):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Stock price tree
    ST = np.array([S * u**j * d**(steps - j) for j in range(steps + 1)])
    
    # Option value at maturity
    if option_type == "call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)
    
    # Step backward through tree
    for i in range(steps - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
    
    return option_values[0]
# =====================
# Streamlit UI Setup
# =====================
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")
st.title("Black-Scholes Option Pricing Dashboard")

# Sidebar Inputs – Primary Parameters
st.sidebar.header("Option Input Parameters")
premium = st.sidebar.slider("Premium", 0.0, 5.0, 2.5, 0.01)
S = st.sidebar.slider("Underlying Price (S)", 0, 1000, 50, 1)
K_input = st.sidebar.slider("Strike Price (K)", 0, 1000, 55, 1)
T = st.sidebar.slider("Time to Expiration (T in years)", 0.01, 10.0, 1.0, 0.01)
r_percent = st.sidebar.slider("Risk-Free Rate (r) [%]", 0.0, 100.0, 2.0, 0.01)
r = r_percent / 100

# Sidebar Inputs – Heatmap Parameters
st.sidebar.header("Heatmap Range Settings")
k_min, k_max = st.sidebar.slider("Strike Price Range (K)", 0, 200, (40, 70), step=1)
k_step = st.sidebar.number_input("Strike Price Step", 1, 50, 5)
v_min, v_max = st.sidebar.slider("Volatility Range (%)", 5, 200, (10, 100), step=5)
v_step = st.sidebar.number_input("Volatility Step (%)", 1, 50, 10)


# Sidebar Inputs – Alternative Model Parameters
st.sidebar.header("Alternative Model Settings")
steps = st.sidebar.slider("Binomial Tree Steps", 10, 500, 100, step=10)
n_simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 100_000, 10000, step=1000)

# =====================
# Calculations for Heatmaps
# =====================
k_values = np.arange(k_min, k_max + k_step, k_step)
vol_values = np.arange(v_min / 100, v_max / 100 + v_step / 100, v_step / 100)

call_matrix = []
put_matrix = []

for vol in vol_values:
    call_row = []
    put_row = []
    for K in k_values:
        call_pnl, put_pnl = calculate_option_prices(S, K, T, r, vol, premium)
        call_row.append(call_pnl)
        put_row.append(put_pnl)
    call_matrix.append(call_row)
    put_matrix.append(put_row)

call_df = pd.DataFrame(
    call_matrix,
    index=[f"{v*100:.0f}%" for v in vol_values],
    columns=[f"{k}" for k in k_values]
)
put_df = pd.DataFrame(
    put_matrix,
    index=[f"{v*100:.0f}%" for v in vol_values],
    columns=[f"{k}" for k in k_values]
)

# =====================
# Input Summary (single row table)
# =====================
summary_df = pd.DataFrame({
    "Premium": [premium],
    "Underlying Price (S)": [S],
    "Strike Price (K)": [K_input],
    "Time to Expiration (T)": [f"{T} years"],
    "Risk-Free Rate (r)": [f"{r:.2%}"]
})

# =====================
# Create Tabs
# =====================
tabs = st.tabs([
    "Overview / Guide",
    "Model Comparison",
    "Heatmaps",
    "Greeks Analysis",
    "Greek Heatmaps",
    "Error Analysis",
    "Other Models",
    "Glossary"
])

# --- Tab 0: Overview / Guide ---
with tabs[0]:
    st.header("Understanding European Options")
    st.info("This dashboard explores **vanilla European call and put options** only. These can be exercised **only at expiration**.")

    st.markdown("""
    ### What is an Option?
    - A **Call Option** gives the holder the right to **buy** an asset at a predetermined price (strike).
    - A **Put Option** gives the holder the right to **sell** an asset at the strike price.

    ### Payoff Profiles:
    """)

    st.image("https://www.researchgate.net/profile/Sanele-Makamo/publication/324123429/figure/fig1/AS:610151379787776@1522482833834/The-illustration-of-payoff-for-standard-options.png")

# --- Tab 1: Model Comparison ---
with tabs[1]:
    st.subheader("Overview / Model Comparison")

    mid_vol = ((v_min + v_max) / 2) / 100
    call_bs, put_bs = calculate_option_prices(S, K_input, T, r, mid_vol, premium)
    call_mc = monte_carlo_option_pricing(S, K_input, T, r, mid_vol, n_simulations=n_simulations, option_type="call") - premium
    put_mc = monte_carlo_option_pricing(S, K_input, T, r, mid_vol, n_simulations=n_simulations, option_type="put") - premium
    call_bin = binomial_option_pricing(S, K_input, T, r, mid_vol, steps=steps, option_type="call") - premium
    put_bin = binomial_option_pricing(S, K_input, T, r, mid_vol, steps=steps, option_type="put") - premium

    comparison_df = pd.DataFrame({
        "Model": ["Black-Scholes", "Monte Carlo", "Binomial Tree"] * 2,
        "Option Type": ["Call"] * 3 + ["Put"] * 3,
        "Price": [call_bs, call_mc, call_bin, put_bs, put_mc, put_bin]
    })

    comparison_df["Price"] = comparison_df["Price"].apply(lambda x: f"{x:.4f}")

    st.dataframe(comparison_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ]), use_container_width=True)

    fig = px.bar(comparison_df, x="Model", y="Price", color="Option Type", barmode="group", title="Option Prices by Model")
    st.plotly_chart(fig)

    st.markdown("""
    - **Black-Scholes**: Closed-form solution, assumes constant volatility and no early exercise.
    - **Monte Carlo**: Good for complex/path-dependent options, less efficient for vanilla.
    - **Binomial Tree**: Flexible for American options; increases accuracy with more steps.
    """)

# --- Tab 2: Heatmaps ---
with tabs[2]:
    st.subheader("Current Input Summary")
    st.dataframe(summary_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Call Heatmap")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.heatmap(call_df, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, linecolor="gray", cbar_kws={'label': 'Call P&L'}, ax=ax1)
        ax1.set_xlabel("Strike Price (K)")
        ax1.set_ylabel("Volatility")
        st.pyplot(fig1)

    with col2:
        st.subheader("📉 Put Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(put_df, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, linecolor="gray", cbar_kws={'label': 'Put P&L'}, ax=ax2)
        ax2.set_xlabel("Strike Price (K)")
        ax2.set_ylabel("Volatility")
        st.pyplot(fig2)

# --- Tab 3: Greeks ---
with tabs[3]:
    st.subheader("Option Greeks (for selected Strike & mid Volatility)")
    greeks = calculate_greeks(S, K_input, T, r, mid_vol)

    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Call": [f"{greeks[g][0]:.4f}" for g in greeks],
        "Put": [f"{greeks[g][1]:.4f}" for g in greeks]
    })
    st.dataframe(greeks_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ]), use_container_width=True)

# --- Tab 4: Greek Heatmaps ---
with tabs[4]:
    st.subheader("Interactive Greek Heatmaps")

    greek_choice = st.selectbox("Select Greek to visualize", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

    greek_matrix = calculate_greek_matrix(S, k_values, T, r, vol_values, greek_choice, premium)
    greek_df = pd.DataFrame(
        greek_matrix,
        index=[f"{v*100:.0f}%" for v in vol_values],
        columns=[f"{k}" for k in k_values]
    )

    fig = px.imshow(
        greek_df,
        labels=dict(x="Strike Price (K)", y="Volatility (%)", color=greek_choice),
        x=greek_df.columns,
        y=greek_df.index,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        origin='lower'
    )
    fig.update_layout(height=600, margin=dict(l=50, r=50, t=50, b=50))
    fig.update_xaxes(side="bottom")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Note:** Values shown are for Call options.")

# --- Tab 5: Error Analysis ---
with tabs[5]:
    st.subheader("Error Analysis vs Black-Scholes")
    comparison_df["Raw Price"] = [call_bs, call_mc, call_bin, put_bs, put_mc, put_bin]
    comparison_df["Error vs BS"] = comparison_df.groupby("Option Type")["Raw Price"].apply(lambda x: ((x - x.iloc[0]) / x.iloc[0]) * 100)

    error_fig = px.bar(
        comparison_df,
        x="Model",
        y="Error vs BS",
        color="Option Type",
        barmode="group",
        title="% Error Compared to Black-Scholes"
    )
    st.plotly_chart(error_fig)

# --- Tab 6: Other Models ---
with tabs[6]:
    st.header("Alternative Option Pricing Models")

    st.markdown("These models provide alternative methods to Black-Scholes, useful especially when assumptions like constant volatility or continuous trading don't hold.")

    st.markdown("Monte Carlo and Binomial results are already integrated into the Model Comparison tab.")

# --- Tab 7: Glossary ---
with tabs[7]:
    st.header("📘 Glossary & Model Overview")
    st.markdown("""
    ### Models

    #### ⬇️ Black-Scholes
    - **Type**: Analytical (Closed-form)
    - **Assumptions**: Constant volatility, no early exercise, lognormal returns
    - **Use**: European vanilla options
    - **Method**: Uses cumulative normal distributions for pricing

    #### 🎲 Monte Carlo
    - **Type**: Stochastic Simulation
    - **Assumptions**: Risk-neutral paths
    - **Use**: Complex or path-dependent payoffs
    - **Method**: Averages discounted simulated payoffs

    #### 🌳 Binomial Tree
    - **Type**: Discrete Lattice
    - **Assumptions**: Stepwise price evolution
    - **Use**: American/European options
    - **Method**: Recursively evaluates payoffs from tree nodes

    ---

    ### Core Terms
    - **Call Option**: Right to buy at strike
    - **Put Option**: Right to sell at strike
    - **Strike Price (K)**: Predetermined exercise price
    - **Underlying Price (S)**: Current price of asset
    - **Volatility (σ)**: Annualized standard deviation of returns
    - **Risk-Free Rate (r)**: Discounting interest rate
    - **Time to Expiry (T)**: In years
    - **Premium**: Cost to purchase the option

    ### Greeks
    - **Delta (Δ)**: Price sensitivity to underlying
    - **Gamma (Γ)**: Rate of change of delta
    - **Vega (ν)**: Sensitivity to volatility
    - **Theta (Θ)**: Time decay per day
    - **Rho (ρ)**: Sensitivity to interest rate
    """)
