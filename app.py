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
tabs = st.tabs(["Heatmaps", "Greeks Analysis", "Glossary", "Greek Heatmaps"])

# --- Tab 1: Heatmaps ---
with tabs[0]:
    st.subheader("Current Input Summary")
    st.dataframe(summary_df.style.set_properties(**{'text-align': 'center'})
                 .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]),
                 use_container_width=True)
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

# --- Tab 2: Greeks ---
with tabs[1]:
    st.subheader("Option Greeks (for selected Strike & mid Volatility)")
    mid_vol = ((v_min + v_max) / 2) / 100
    greeks = calculate_greeks(S, K_input, T, r, mid_vol)

    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Call": [f"{greeks[g][0]:.4f}" for g in greeks],
        "Put": [f"{greeks[g][1]:.4f}" for g in greeks]
    })
    st.dataframe(greeks_df.style.set_properties(**{'text-align': 'center'})
                 .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]),
                 use_container_width=True)

# --- Tab 3: Glossary ---
with tabs[2]:
    st.header("Glossary of Terms")
    st.markdown("""
    ### Black-Scholes Model Inputs
    - **Premium:** The cost paid upfront for the option.
    - **Underlying Price (S):** Current price of the underlying asset.
    - **Strike Price (K):** The price at which the option can be exercised.
    - **Time to Expiration (T):** Time left until the option expires (in years).
    - **Risk-Free Rate (r):** Annual risk-free interest rate as a decimal.
    - **Volatility (σ):** Annualized standard deviation of the underlying asset's returns.

    ### Option Greeks
    - **Delta (Δ):** Sensitivity of option price to a $1 change in underlying asset price.
    - **Gamma (Γ):** Rate of change of Delta with respect to the underlying price.
    - **Vega (ν):** Sensitivity to volatility; change in option price for a 1% change in volatility.
    - **Theta (Θ):** Time decay; change in option price for a one-day decrease in time to expiry.
    - **Rho (ρ):** Sensitivity to interest rates; change in option price for a 1% change in risk-free rate.
    """)

with tabs[3]:
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
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    fig.update_xaxes(side="bottom")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Note:** Values shown are for Call options.")

