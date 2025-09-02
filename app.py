import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =====================
# Enhanced Black-Scholes with Real Market Data
# =====================

class BlackScholesEngine:
    """Professional Black-Scholes pricing engine with advanced features"""
    
    @staticmethod
    def d1_d2(S, K, T, r, vol):
        """Calculate d1 and d2 parameters"""
        if T <= 0 or vol <= 0:
            return 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def option_prices(S, K, T, r, vol, dividend_yield=0):
        """Calculate option prices with dividend adjustment"""
        d1, d2 = BlackScholesEngine.d1_d2(S, K, T, r, vol)
        
        call_price = (S * np.exp(-dividend_yield * T) * norm.cdf(d1) - 
                     K * np.exp(-r * T) * norm.cdf(d2))
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                    S * np.exp(-dividend_yield * T) * norm.cdf(-d1))
        
        return call_price, put_price
    
    @staticmethod
    def greeks(S, K, T, r, vol, dividend_yield=0):
        """Calculate all Greeks"""
        if T <= 0 or vol <= 0:
            return {greek: (0, 0) for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho", "Charm", "Vanna"]}
            
        d1, d2 = BlackScholesEngine.d1_d2(S, K, T, r, vol)
        
        # First-order Greeks
        delta_call = np.exp(-dividend_yield * T) * norm.cdf(d1)
        delta_put = np.exp(-dividend_yield * T) * (norm.cdf(d1) - 1)
        
        # Second-order Greeks
        gamma = (np.exp(-dividend_yield * T) * norm.pdf(d1)) / (S * vol * np.sqrt(T))
        vega = (S * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T)) / 100
        
        # Theta (time decay)
        theta_call = ((-S * np.exp(-dividend_yield * T) * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - 
                      r * K * np.exp(-r * T) * norm.cdf(d2) + 
                      dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(d1)) / 365)
        
        theta_put = ((-S * np.exp(-dividend_yield * T) * norm.pdf(d1) * vol / (2 * np.sqrt(T)) + 
                     r * K * np.exp(-r * T) * norm.cdf(-d2) - 
                     dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)) / 365)
        
        # Rho (interest rate sensitivity)
        rho_call = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        rho_put = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
        
        # Second-order cross Greeks
        charm_call = (-np.exp(-dividend_yield * T) * norm.pdf(d1) * 
                     (2 * (r - dividend_yield) * T - d2 * vol * np.sqrt(T)) / 
                     (2 * T * vol * np.sqrt(T))) / 365
        
        vanna = (vega / S) * (1 - d1 / (vol * np.sqrt(T)))
        
        return {
            "Delta": (delta_call, delta_put),
            "Gamma": (gamma, gamma),
            "Vega": (vega, vega),
            "Theta": (theta_call, theta_put),
            "Rho": (rho_call, rho_put),
            "Charm": (charm_call, charm_call),
            "Vanna": (vanna, vanna)
        }
    
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call', dividend_yield=0, max_iter=100):
        """Calculate implied volatility using Newton-Raphson method"""
        vol = 0.3  # Initial guess
        tolerance = 1e-6
        
        for i in range(max_iter):
            call_price, put_price = BlackScholesEngine.option_prices(S, K, T, r, vol, dividend_yield)
            price = call_price if option_type == 'call' else put_price
            
            vega = BlackScholesEngine.greeks(S, K, T, r, vol, dividend_yield)["Vega"][0] * 100
            
            if abs(vega) < 1e-10:
                break
                
            vol_new = vol - (price - market_price) / vega
            
            if abs(vol_new - vol) < tolerance:
                return max(vol_new, 0.001)
                
            vol = max(vol_new, 0.001)
        
        return vol

def fetch_market_data(symbol):
    """Fetch real market data for analysis"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        info = stock.info
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate realized volatility
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        realized_vol = returns.std() * np.sqrt(252)
        
        return {
            'current_price': current_price,
            'realized_volatility': realized_vol,
            'price_history': hist,
            'company_name': info.get('longName', symbol)
        }
    except:
        return None

def calculate_portfolio_greeks(positions):
    """Calculate portfolio-level Greeks"""
    portfolio_greeks = {greek: 0 for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}
    
    for pos in positions:
        greeks = BlackScholesEngine.greeks(
            pos['S'], pos['K'], pos['T'], pos['r'], pos['vol']
        )
        
        multiplier = pos['quantity'] * (1 if pos['type'] == 'call' else 1)
        greek_idx = 0 if pos['type'] == 'call' else 1
        
        for greek in portfolio_greeks:
            portfolio_greeks[greek] += greeks[greek][greek_idx] * multiplier
    
    return portfolio_greeks

# =====================
# Streamlit Configuration
# =====================
st.set_page_config(
    page_title="Professional Options Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏛️ Professional Options Analytics Platform</div>', unsafe_allow_html=True)

# =====================
# Enhanced Sidebar
# =====================
st.sidebar.markdown("### 📊 Market Parameters")

# Real market data integration
use_real_data = st.sidebar.checkbox("Use Real Market Data", value=False)
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", disabled=not use_real_data)

market_data = None
if use_real_data and symbol:
    market_data = fetch_market_data(symbol)
    if market_data:
        st.sidebar.success(f"✅ Data loaded for {market_data['company_name']}")
        default_price = market_data['current_price']
        default_vol = market_data['realized_volatility']
    else:
        st.sidebar.error("❌ Failed to fetch market data")
        default_price = 100
        default_vol = 0.2
else:
    default_price = 100
    default_vol = 0.2

# Enhanced parameters
S = st.sidebar.slider("Underlying Price ($)", 10.0, 500.0, float(default_price), 0.1)
K_input = st.sidebar.slider("Strike Price ($)", 10.0, 500.0, float(default_price * 1.1), 0.1)
T = st.sidebar.slider("Time to Expiration (days)", 1, 365, 30) / 365
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.01) / 100
vol_input = st.sidebar.slider("Volatility (%)", 5.0, 100.0, default_vol * 100, 0.1) / 100
dividend_yield = st.sidebar.slider("Dividend Yield (%)", 0.0, 5.0, 0.0, 0.01) / 100

# Advanced settings
st.sidebar.markdown("### ⚙️ Advanced Settings")
greek_sensitivity = st.sidebar.slider("Greek Sensitivity Analysis Range (±%)", 5, 50, 20)
heatmap_resolution = st.sidebar.selectbox("Heatmap Resolution", ["Low (15x15)", "Medium (25x25)", "High (35x35)"], index=1)

# Determine grid size based on resolution
grid_sizes = {"Low (15x15)": 15, "Medium (25x25)": 25, "High (35x35)": 35}
grid_size = grid_sizes[heatmap_resolution]

# =====================
# Main Tabs
# =====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Market Overview", 
    "🎯 Option Pricing", 
    "📊 Greeks Analysis", 
    "🔥 Risk Heatmaps", 
    "📉 P&L Scenarios", 
    "💼 Portfolio Analytics"
])

# --- Tab 1: Market Overview ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    call_price, put_price = BlackScholesEngine.option_prices(S, K_input, T, r, vol_input, dividend_yield)
    
    with col1:
        st.metric("Call Price", f"${call_price:.2f}")
    with col2:
        st.metric("Put Price", f"${put_price:.2f}")
    with col3:
        st.metric("Call-Put Parity Check", f"${call_price - put_price + K_input * np.exp(-r * T) - S:.2f}")
    with col4:
        st.metric("Intrinsic Value (Call)", f"${max(S - K_input, 0):.2f}")
    
    if market_data:
        st.markdown("### 📈 Price History")
        fig = px.line(
            x=market_data['price_history'].index,
            y=market_data['price_history']['Close'],
            title=f"{market_data['company_name']} Stock Price"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Option Pricing ---
with tab2:
    st.markdown("### 🎯 Advanced Option Pricing Analysis")
    
    # Moneyness analysis
    moneyness = S / K_input
    st.markdown(f"**Current Moneyness:** {moneyness:.3f} {'(ITM)' if moneyness > 1 else '(OTM)' if moneyness < 1 else '(ATM)'}")
    
    col1, col2 = st.columns(2)
    
    # Price sensitivity to underlying
    with col1:
        st.markdown("#### Price vs Underlying")
        spot_range = np.linspace(S * 0.7, S * 1.3, 50)
        call_prices = [BlackScholesEngine.option_prices(s, K_input, T, r, vol_input, dividend_yield)[0] for s in spot_range]
        put_prices = [BlackScholesEngine.option_prices(s, K_input, T, r, vol_input, dividend_yield)[1] for s in spot_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=call_prices, name="Call", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=spot_range, y=put_prices, name="Put", line=dict(color='red')))
        fig.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
        fig.add_vline(x=K_input, line_dash="dot", annotation_text="Strike Price")
        fig.update_layout(title="Option Value vs Underlying Price", xaxis_title="Spot Price", yaxis_title="Option Value")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time decay analysis
    with col2:
        st.markdown("#### Time Decay Analysis")
        time_range = np.linspace(0.01, T, 30)
        call_time_values = [BlackScholesEngine.option_prices(S, K_input, t, r, vol_input, dividend_yield)[0] for t in time_range]
        put_time_values = [BlackScholesEngine.option_prices(S, K_input, t, r, vol_input, dividend_yield)[1] for t in time_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_range*365, y=call_time_values, name="Call", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_range*365, y=put_time_values, name="Put", line=dict(color='orange')))
        fig.update_layout(title="Time Decay", xaxis_title="Days to Expiration", yaxis_title="Option Value")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Greeks Analysis ---
with tab3:
    st.markdown("### 📊 Comprehensive Greeks Analysis")
    
    greeks = BlackScholesEngine.greeks(S, K_input, T, r, vol_input, dividend_yield)
    
    # Greeks dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Delta (Call)", f"{greeks['Delta'][0]:.4f}")
        st.metric("Gamma", f"{greeks['Gamma'][0]:.6f}")
    with col2:
        st.metric("Delta (Put)", f"{greeks['Delta'][1]:.4f}")
        st.metric("Vega", f"{greeks['Vega'][0]:.4f}")
    with col3:
        st.metric("Theta (Call)", f"{greeks['Theta'][0]:.4f}")
        st.metric("Rho (Call)", f"{greeks['Rho'][0]:.4f}")
    with col4:
        st.metric("Theta (Put)", f"{greeks['Theta'][1]:.4f}")
        st.metric("Charm", f"{greeks['Charm'][0]:.6f}")
    
    # Greeks surfaces
    st.markdown("#### Greeks Sensitivity Analysis")
    
    # Create ranges for sensitivity analysis
    spot_range = np.linspace(S * (1 - greek_sensitivity/100), S * (1 + greek_sensitivity/100), 20)
    vol_range = np.linspace(vol_input * 0.5, vol_input * 1.5, 20)
    
    # Calculate Delta surface
    delta_surface = np.zeros((len(vol_range), len(spot_range)))
    gamma_surface = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            greeks_temp = BlackScholesEngine.greeks(spot, K_input, T, r, vol, dividend_yield)
            delta_surface[i, j] = greeks_temp['Delta'][0]
            gamma_surface[i, j] = greeks_temp['Gamma'][0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Surface(z=delta_surface, x=spot_range, y=vol_range*100, colorscale="RdYlBu")])
        fig.update_layout(title="Delta Surface", scene=dict(
            xaxis_title="Spot Price", 
            yaxis_title="Volatility (%)", 
            zaxis_title="Delta"
        ), height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Surface(z=gamma_surface, x=spot_range, y=vol_range*100, colorscale="Plasma")])
        fig.update_layout(title="Gamma Surface", scene=dict(
            xaxis_title="Spot Price", 
            yaxis_title="Volatility (%)", 
            zaxis_title="Gamma"
        ), height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Risk Heatmaps ---
with tab4:
    st.markdown("### 🔥 Advanced Risk Analysis")
    
    # High-resolution heatmaps
    strike_range = np.linspace(S * 0.8, S * 1.2, grid_size)
    vol_range_hm = np.linspace(0.1, 0.8, grid_size)
    
    # Calculate P&L surfaces
    call_pnl_matrix = np.zeros((len(vol_range_hm), len(strike_range)))
    put_pnl_matrix = np.zeros((len(vol_range_hm), len(strike_range)))
    
    premium_paid_call = call_price
    premium_paid_put = put_price
    
    for i, vol in enumerate(vol_range_hm):
        for j, strike in enumerate(strike_range):
            current_call, current_put = BlackScholesEngine.option_prices(S, strike, T, r, vol, dividend_yield)
            call_pnl_matrix[i, j] = current_call - premium_paid_call
            put_pnl_matrix[i, j] = current_put - premium_paid_put
    
    # Interactive heatmaps
    option_type = st.selectbox("Select Analysis Type", ["Call P&L", "Put P&L", "Portfolio VaR"])
    
    if option_type == "Call P&L":
        fig = px.imshow(
            call_pnl_matrix, 
            x=strike_range.round(2), 
            y=(vol_range_hm * 100).round(1),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Call Option P&L Heatmap"
        )
        fig.update_xaxis(title="Strike Price")
        fig.update_yaxis(title="Volatility (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif option_type == "Put P&L":
        fig = px.imshow(
            put_pnl_matrix, 
            x=strike_range.round(2), 
            y=(vol_range_hm * 100).round(1),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Put Option P&L Heatmap"
        )
        fig.update_xaxis(title="Strike Price")
        fig.update_yaxis(title="Volatility (%)")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: P&L Scenarios ---
with tab5:
    st.markdown("### 📉 Scenario Analysis & Risk Management")
    
    # Monte Carlo simulation
    st.markdown("#### Monte Carlo Price Simulation")
    
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)
    time_horizon = st.slider("Time Horizon (days)", 1, 30, 7)
    
    # Generate price paths
    dt = time_horizon / 365
    np.random.seed(42)  # For reproducibility
    
    price_paths = []
    for _ in range(num_simulations):
        random_shock = np.random.normal(0, 1)
        future_price = S * np.exp((r - dividend_yield - 0.5 * vol_input**2) * dt + 
                                 vol_input * np.sqrt(dt) * random_shock)
        price_paths.append(future_price)
    
    price_paths = np.array(price_paths)
    
    # Calculate option payoffs
    call_payoffs = np.maximum(price_paths - K_input, 0)
    put_payoffs = np.maximum(K_input - price_paths, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(price_paths, nbins=50, title="Simulated Price Distribution")
        fig.add_vline(x=S, line_dash="dash", annotation_text="Current Price", line_color="red")
        fig.add_vline(x=K_input, line_dash="dot", annotation_text="Strike Price", line_color="blue")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics
        var_95 = np.percentile(call_payoffs, 5)
        cvar_95 = np.mean(call_payoffs[call_payoffs <= var_95])
        
        st.markdown("#### Risk Metrics")
        st.metric("95% VaR (Call)", f"${var_95:.2f}")
        st.metric("95% CVaR (Call)", f"${cvar_95:.2f}")
        st.metric("Expected Payoff (Call)", f"${np.mean(call_payoffs):.2f}")
        st.metric("Probability ITM (Call)", f"{(np.sum(price_paths > K_input) / num_simulations * 100):.1f}%")

# --- Tab 6: Portfolio Analytics ---
with tab6:
    st.markdown("### 💼 Portfolio-Level Analytics")
    
    st.markdown("#### Position Builder")
    
    # Simple portfolio builder
    if 'portfolio_positions' not in st.session_state:
        st.session_state.portfolio_positions = []
    
    with st.expander("Add New Position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pos_type = st.selectbox("Type", ["call", "put"], key="pos_type")
        with col2:
            pos_strike = st.number_input("Strike", value=K_input, key="pos_strike")
        with col3:
            pos_quantity = st.number_input("Quantity", value=100, key="pos_qty")
        with col4:
            pos_action = st.selectbox("Action", ["Buy", "Sell"], key="pos_action")
        
        if st.button("Add Position"):
            new_position = {
                'type': pos_type,
                'strike': pos_strike,
                'quantity': pos_quantity * (1 if pos_action == "Buy" else -1),
                'S': S, 'K': pos_strike, 'T': T, 'r': r, 'vol': vol_input
            }
            st.session_state.portfolio_positions.append(new_position)
    
    # Display current positions
    if st.session_state.portfolio_positions:
        st.markdown("#### Current Portfolio")
        
        portfolio_df = pd.DataFrame([
            {
                'Type': pos['type'].title(),
                'Strike': f"${pos['strike']:.2f}",
                'Quantity': pos['quantity'],
                'Current Value': f"${BlackScholesEngine.option_prices(S, pos['strike'], T, r, vol_input)[0 if pos['type'] == 'call' else 1] * abs(pos['quantity']) / 100:.2f}"
            }
            for pos in st.session_state.portfolio_positions
        ])
        
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Portfolio Greeks
        portfolio_greeks = calculate_portfolio_greeks(st.session_state.portfolio_positions)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Portfolio Delta", f"{portfolio_greeks['Delta']:.2f}")
        with col2:
            st.metric("Portfolio Gamma", f"{portfolio_greeks['Gamma']:.4f}")
        with col3:
            st.metric("Portfolio Vega", f"{portfolio_greeks['Vega']:.2f}")
        with col4:
            st.metric("Portfolio Theta", f"{portfolio_greeks['Theta']:.2f}")
        with col5:
            st.metric("Portfolio Rho", f"{portfolio_greeks['Rho']:.2f}")
        
        if st.button("Clear Portfolio"):
            st.session_state.portfolio_positions = []
            st.rerun()

# =====================
# Footer
# =====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Professional Options Analytics Platform</strong></p>
    <p>Built for quantitative analysis and risk management • Real-time market data integration</p>
    <p>Advanced Greeks calculation • Monte Carlo simulation • Portfolio optimization</p>
</div>
""", unsafe_allow_html=True)
