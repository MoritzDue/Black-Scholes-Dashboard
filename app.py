import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

# =====================
# Option Pricing & Greeks
# =====================
def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

def calculate_greeks(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1

    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change
    theta_call = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "Delta": (delta_call, delta_put),
        "Gamma": (gamma, gamma),
        "Vega": (vega, vega),
        "Theta": (theta_call, theta_put),
        "Rho": (rho_call, rho_put)
    }

# =====================
# Streamlit UI
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
# Calculations
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

call_df = pd.DataFrame(call_matrix
