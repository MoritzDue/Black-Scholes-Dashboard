import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go   # <-- changed here

# =====================
# Black-Scholes Pricing & Greeks
# =====================
def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

def calculate_greek_matrix(S, K_vals, T, r, vol_vals, greek_name):
    matrix = []
    for vol in vol_vals:
        row = []
        for K in K_vals:
            d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)

            delta_call = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            theta_call = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

            greek_map = {
                "Delta": delta_call,
                "Gamma": gamma,
                "Vega": vega,
                "Theta": theta_call,
                "Rho": rho_call,
            }
            row.append(greek_map[greek_name])
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
# Streamlit Setup
# =====================
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")
st.title("Black-Scholes Option Pricing Dashboard")

# Sidebar Inputs
st.sidebar.header("Option Input Parameters")
premium = st.sidebar.slider("Premium", 0.0, 5.0, 2.5, 0.01)
S = st.sidebar.slider("Underlying Price (S)", 0, 1000, 50, 1)
K_input = st.sidebar.slider("Strike Price (K)", 0, 1000, 55, 1)
T = st.sidebar.slider("Time to Expiration (T in years)", 0.01, 10.0, 1.0, 0.01)
r_percent = st.sidebar.slider("Risk-Free Rate (r) [%]", 0.0, 100.0, 2.0, 0.01)
r = r_percent / 100

# Heatmap Ranges
st.sidebar.header("Heatmap Range Settings")
k_min, k_max = st.sidebar.slider("Strike Price Range (K)", 0, 200, (40, 70), step=1)
k_step = st.sidebar.number_input("Strike Price Step", 1, 50, 5)
v_min, v_max = st.sidebar.slider("Volatility Range (%)", 5, 200, (10, 100), step=5)
v_step = st.sidebar.number_input("Volatility Step (%)", 1, 50, 10)

# Calculation grids
k_values = np.arange(k_min, k_max + k_step, k_step)
vol_values = np.arange(v_min / 100, v_max / 100 + v_step / 100, v_step / 100)

# Option matrices
call_matrix, put_matrix = [], []
for vol in vol_values:
    call_row, put_row = [], []
    for K in k_values:
        call_pnl, put_pnl = calculate_option_prices(S, K, T, r, vol, premium)
        call_row.append(call_pnl)
        put_row.append(put_pnl)
    call_matrix.append(call_row)
    put_matrix.append(put_row)

call_matrix = np.array(call_matrix)
put_matrix = np.array(put_matrix)

call_df = pd.DataFrame(call_matrix, index=[f"{v*100:.0f}%" for v in vol_values], columns=[f"{k}" for k in k_values])
put_df = pd.DataFrame(put_matrix, index=[f"{v*100:.0f}%" for v in vol_values], columns=[f"{k}" for k in k_values])

# =====================
# Tabs
# =====================
tabs = st.tabs(["Overview", "Option Surfaces", "Heatmaps", "Greeks"])

# --- Tab 0: Overview ---
with tabs[0]:
    st.header("Understanding European Options")
    st.info("We explore **vanilla European call and put options** using the Black-Scholes model.")
    st.image("https://www.researchgate.net/profile/Sanele-Makamo/publication/324123429/figure/fig1/AS:610151379787776@1522482833834/The-illustration-of-payoff-for-standard-options.png")

# --- Tab 1: Option Surfaces ---
with tabs[1]:
    st.subheader("3D Surfaces for Option Value")

    fig_call = go.Figure(data=[go.Surface(
        z=call_matrix, x=k_values, y=vol_values*100, colorscale="Viridis"
    )])
    fig_call.update_layout(title="Call Option Surface",
        scene=dict(xaxis_title="Strike (K)", yaxis_title="Volatility (%)", zaxis_title="Call Value"))
    st.plotly_chart(fig_call, use_container_width=True)

    fig_put = go.Figure(data=[go.Surface(
        z=put_matrix, x=k_values, y=vol_values*100, colorscale="Cividis"
    )])
    fig_put.update_layout(title="Put Option Surface",
        scene=dict(xaxis_title="Strike (K)", yaxis_title="Volatility (%)", zaxis_title="Put Value"))
    st.plotly_chart(fig_put, use_container_width=True)

# --- Tab 2: Heatmaps ---
with tabs[2]:
    st.subheader("Option Heatmaps")
    option_choice = st.selectbox("Select Option Type", ["Call", "Put"])
    if option_choice == "Call":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(call_df, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.5, linecolor="gray",
                    cbar_kws={'label': 'Call P&L'}, ax=ax)
        ax.set_xlabel("Strike Price (K)")
        ax.set_ylabel("Volatility")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(put_df, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.5, linecolor="gray",
                    cbar_kws={'label': 'Put P&L'}, ax=ax)
        ax.set_xlabel("Strike Price (K)")
        ax.set_ylabel("Volatility")
        st.pyplot(fig)

# --- Tab 3: Greeks ---
with tabs[3]:
    st.subheader("Option Greeks")
    mid_vol = ((v_min + v_max) / 2) / 100
    greeks = calculate_greeks(S, K_input, T, r, mid_vol)

    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Call": [f"{greeks[g][0]:.4f}" for g in greeks],
        "Put": [f"{greeks[g][1]:.4f}" for g in greeks]
    })
    st.dataframe(greeks_df, use_container_width=True)

    st.subheader("3D Greek Surfaces")
    greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
    greek_matrix = calculate_greek_matrix(S, k_values, T, r, vol_values, greek_choice)

    fig_greek = go.Figure(data=[go.Surface(
        z=greek_matrix, x=k_values, y=vol_values*100, colorscale="Plasma"
    )])
    fig_greek.update_layout(title=f"{greek_choice} Surface",
        scene=dict(xaxis_title="Strike (K)", yaxis_title="Volatility (%)", zaxis_title=greek_choice))
    st.plotly_chart(fig_greek, use_container_width=True)
