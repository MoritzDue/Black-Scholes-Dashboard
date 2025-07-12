import streamlit as st
import numpy as np
from scipy.stats import norm

def calculate_option_prices(S, K, T, r, vol, premium):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - premium
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - premium
    return C, P

# Streamlit app
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="centered")
st.title("Black-Scholes Option Pricing Dashboard")

st.sidebar.header("Input Parameters")

premium = st.sidebar.slider("Premium", 0.0, 5.0, 2.5, 0.01)
S = st.sidebar.slider("Underlying Price (S)", 0, 1000, 50, 1)
K = st.sidebar.slider("Strike Price (K)", 0, 1000, 55, 1)
T = st.sidebar.slider("Time to Expiration (T in years)", 0.01, 10.0, 1.0, 0.01)
r_percent = st.sidebar.slider("Risk-Free Rate (r) [%]", 0.0, 100.0, 2.0, 0.01)
r = r_percent / 100  # Convert to decimal

if st.button("Calculate Option Prices"):
    st.subheader("Inputs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Premium", f"{premium}")
    col2.metric("Underlying Price (S)", f"{S}")
    col3.metric("Strike Price (K)", f"{K}")

    col4, col5 = st.columns(2)
    col4.metric("Time to Expiry (T)", f"{T} years")
    col5.metric("Risk-Free Rate (r)", f"{r:.2%}")

    st.subheader("Call & Put P&L by Volatility")
    data = {
        "Volatility": [],
        "Call P&L": [],
        "Put P&L": [],
    }

    for i in range(1, 11):
        vol = i * 0.10
        C, P = calculate_option_prices(S, K, T, r, vol, premium)
        data["Volatility"].append(round(vol, 2))
        data["Call P&L"].append(round(C, 2))
        data["Put P&L"].append(round(P, 2))

    st.table(data)
