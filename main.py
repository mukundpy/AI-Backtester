import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn.linear_model as lm

from data_loader import load_stock_data, POPULAR_NSE_STOCKS, format_ticker_symbol
from indicators import add_indicators
from volatility_model import fit_garch_volatility
from strategies import generate_signals, STRATEGY_OPTIONS
from ml_model import train_and_predict_ml_signals
from lstm_model import train_and_predict_lstm_signals
from walk_forward import run_walk_forward_validation
from robustness import run_multi_stock_robustness_test
from backtester import run_backtest
from metrics import calculate_metrics, build_model_comparison_table

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #2a2e39;
    }
    .metric-title {
        font-size: 13px;
        color: #848e9c;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
    }
    .positive-val { color: #0ecb81; }
    .negative-val { color: #f6465d; }
    .neutral-val { color: #f0b90b; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Trading Strategy Backtester")
st.caption("Simulate rule-based and AI-driven hybrid model strategies (Random Forest, XGBoost, GARCH Volatility Layer, LSTM Benchmark, and Walk-Forward Validation) on Indian (NSE) equities with zero look-ahead bias.")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Stock & Timeframe")

stock_mode = st.sidebar.radio("Select Input Mode:", ["Popular NSE Stocks", "Custom Ticker"], index=0)
if stock_mode == "Popular NSE Stocks":
    selected_stock_label = st.sidebar.selectbox(
        "Choose Stock:",
        options=list(POPULAR_NSE_STOCKS.keys()),
        format_func=lambda x: f"{x} - {POPULAR_NSE_STOCKS[x]}"
    )
    ticker_input = selected_stock_label
else:
    ticker_input = st.sidebar.text_input("Enter Ticker (e.g. RELIANCE or TCS):", value="RELIANCE")

ticker_symbol = format_ticker_symbol(ticker_input)

period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
selected_period = st.sidebar.selectbox("Historical Period:", period_options, index=3)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Configuration")

strategy_choice = st.sidebar.selectbox("Select Strategy:", STRATEGY_OPTIONS, index=3)

strategy_params = {}

if strategy_choice == "MA Crossover":
    st.sidebar.subheader("MA Parameters")
    strategy_params["fast_period"] = st.sidebar.slider("Fast Moving Average (days):", min_value=5, max_value=50, value=20)
    strategy_params["slow_period"] = st.sidebar.slider("Slow Moving Average (days):", min_value=20, max_value=200, value=50)
    
elif strategy_choice == "RSI Threshold":
    st.sidebar.subheader("RSI Parameters")
    strategy_params["rsi_period"] = st.sidebar.slider("RSI Period:", min_value=5, max_value=30, value=14)
    strategy_params["oversold"] = st.sidebar.slider("Oversold Level (Buy):", min_value=10, max_value=40, value=30)
    strategy_params["overbought"] = st.sidebar.slider("Overbought Level (Sell):", min_value=60, max_value=90, value=70)
    
elif strategy_choice == "Bollinger Reversion":
    st.sidebar.subheader("Bollinger Parameters")
    strategy_params["bb_period"] = st.sidebar.slider("Window Period:", min_value=10, max_value=50, value=20)
    strategy_params["bb_std"] = st.sidebar.slider("Std Dev Multiplier:", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    
else:
    st.sidebar.subheader("AI Model & Hybrid Parameters")
    strategy_params["horizon"] = st.sidebar.slider("Prediction Horizon (days):", min_value=1, max_value=10, value=5)
    strategy_params["confidence"] = st.sidebar.slider("Buy Confidence Threshold:", min_value=0.50, max_value=0.80, value=0.55, step=0.01)
    if "LSTM" in strategy_choice:
        strategy_params["seq_length"] = st.sidebar.slider("LSTM Sequence Window (days):", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk & Volatility Settings")

initial_capital = st.sidebar.number_input("Initial Capital (₹):", min_value=10000, max_value=10000000, value=100000, step=10000)
position_size_pct = st.sidebar.slider("Position Size (% of Capital):", min_value=10, max_value=100, value=100, step=10) / 100.0
transaction_fee_pct = st.sidebar.number_input("Transaction Cost (% per trade):", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100.0
slippage_pct = st.sidebar.number_input("Slippage (% per trade):", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100.0

enable_garch_risk = st.sidebar.checkbox("Enable Dynamic GARCH Risk Management", value=True)
garch_multiplier = st.sidebar.slider("GARCH Band / Risk Multiplier (k):", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
strategy_params["garch_multiplier"] = garch_multiplier

enable_stop_loss = st.sidebar.checkbox("Enable Static Stop-Loss", value=False)
stop_loss_val = st.sidebar.slider("Static Stop-Loss (%):", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100.0 if enable_stop_loss else None

chart_type = st.sidebar.selectbox("Price Chart Style:", ["Candlestick", "Line"], index=0)

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
try:
    with st.spinner(f"Loading data for {ticker_symbol}..."):
        raw_df, stock_info = load_stock_data(ticker_symbol, period=selected_period)
        
    featured_df = add_indicators(
        raw_df,
        fast_ma=strategy_params.get("fast_period", 20),
        slow_ma=strategy_params.get("slow_period", 50),
        rsi_period=strategy_params.get("rsi_period", 14),
        bb_period=strategy_params.get("bb_period", 20),
        bb_std=strategy_params.get("bb_std", 2.0)
    )
    
    featured_df, garch_meta = fit_garch_volatility(featured_df, multiplier=garch_multiplier)
    
    # Section 11 Error Handling Check
    if stock_info.get("num_bars", 0) < 50 and "AI" in strategy_choice:
        st.warning("⚠️ Insufficient historical data points (<50 bars) for machine learning training. Switched to rule-based strategy.")
        strategy_choice = "MA Crossover"
        
    signals, strat_meta = generate_signals(raw_df, strategy_choice, strategy_params)
    
    backtest_res = run_backtest(
        featured_df,
        signals,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        transaction_fee_pct=transaction_fee_pct,
        slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_val,
        use_garch_risk=enable_garch_risk
    )
    
    metrics_res = calculate_metrics(
        backtest_res["equity_curve"],
        backtest_res["trade_log"],
        initial_capital=initial_capital
    )
    
except Exception as e:
    st.error(f"⚠️ Error fetching data or initializing backtester: {str(e)}")
    if st.button("🔄 Retry Data Fetch"):
        st.experimental_rerun()
    st.stop()

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_tech, tab_backtest, tab_robustness, tab_forecast, tab_data = st.tabs([
    "📊 Overview",
    "📈 Technical Analysis",
    "⚡ Backtest Results",
    "🛡️ Robustness & Validation",
    "🤖 AI Forecast",
    "📁 Data & Export"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    company_name = stock_info.get("shortName", ticker_symbol)
    latest_price = featured_df['Close'].iloc[-1]
    prev_close = featured_df['Close'].iloc[-2] if len(featured_df) > 1 else latest_price
    price_change = latest_price - prev_close
    price_change_pct = (price_change / prev_close) * 100.0
    
    if stock_info.get("data_gap_filled", False):
        st.info("ℹ️ **Data Resiliency Note**: Non-trading gaps or missing API data points were automatically forward-filled (`ffill()`) for historical accuracy.")
        
    st.subheader(f"{company_name} ({ticker_symbol})")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color_cls = "positive-val" if price_change >= 0 else "negative-val"
        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Price</div>
            <div class="metric-value {color_cls}">₹{latest_price:,.2f}</div>
            <div style="font-size:12px;" class="{color_cls}">{price_change:+.2f} ({price_change_pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">52-Week High</div>
            <div class="metric-value neutral-val">₹{stock_info.get('fiftyTwoWeekHigh', 0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">52-Week Low</div>
            <div class="metric-value neutral-val">₹{stock_info.get('fiftyTwoWeekLow', 0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        garch_ann_vol = garch_meta.get("latest_ann_vol_pct", 0.0)
        c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">GARCH(1,1) Ann. Volatility</div>
            <div class="metric-value neutral-val">{garch_ann_vol:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(
        x=featured_df.index, y=featured_df['Close'],
        mode='lines', name='Close Price', line=dict(color='#2962FF', width=2)
    ))
    fig_overview.update_layout(
        title=f"Historical Price Movement ({selected_period})",
        xaxis_title="Date", yaxis_title="Price (₹)",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_overview, use_container_width=True)

# =============================================================================
# TAB 2: TECHNICAL ANALYSIS
# =============================================================================
with tab_tech:
    st.subheader("Technical Indicators & GARCH Volatility Bands")
    
    show_garch_bands = st.checkbox("Overlay GARCH(1,1) Forecasted Volatility Bands", value=True)
    show_bollinger_bands = st.checkbox("Overlay Standard Bollinger Bands", value=False)
    
    fig_tech = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("Price & Volatility Overlays", "RSI (14)", "Volume")
    )
    
    if chart_type == "Candlestick":
        fig_tech.add_trace(go.Candlestick(
            x=featured_df.index,
            open=featured_df['Open'], high=featured_df['High'],
            low=featured_df['Low'], close=featured_df['Close'],
            name='OHLC'
        ), row=1, col=1)
    else:
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['Close'],
            mode='lines', name='Close Price', line=dict(color='#FFFFFF', width=1.5)
        ), row=1, col=1)
        
    if 'MA_Fast' in featured_df.columns:
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['MA_Fast'],
            mode='lines', name=f'MA Fast ({strategy_params.get("fast_period", 20)})',
            line=dict(color='#FFB300', width=1.5)
        ), row=1, col=1)
        
    if 'MA_Slow' in featured_df.columns:
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['MA_Slow'],
            mode='lines', name=f'MA Slow ({strategy_params.get("slow_period", 50)})',
            line=dict(color='#E91E63', width=1.5)
        ), row=1, col=1)
        
    if show_bollinger_bands and 'BB_Upper' in featured_df.columns:
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['BB_Upper'],
            mode='lines', name='BB Upper', line=dict(color='#26A69A', width=1, dash='dash')
        ), row=1, col=1)
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['BB_Lower'],
            mode='lines', name='BB Lower', line=dict(color='#26A69A', width=1, dash='dash')
        ), row=1, col=1)
        
    if show_garch_bands and 'GARCH_Upper' in featured_df.columns:
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['GARCH_Upper'],
            mode='lines', name=f'GARCH Upper ({garch_multiplier}σ)',
            line=dict(color='#FF007F', width=1.2, dash='solid')
        ), row=1, col=1)
        fig_tech.add_trace(go.Scatter(
            x=featured_df.index, y=featured_df['GARCH_Lower'],
            mode='lines', name=f'GARCH Lower ({garch_multiplier}σ)',
            line=dict(color='#FF007F', width=1.2, dash='solid'),
            fill='tonexty', fillcolor='rgba(255, 0, 127, 0.05)'
        ), row=1, col=1)
        
    fig_tech.add_trace(go.Scatter(
        x=featured_df.index, y=featured_df['RSI'],
        mode='lines', name='RSI', line=dict(color='#AB47BC', width=1.5)
    ), row=2, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="#EF5350", row=2, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="#26A69A", row=2, col=1)
    
    colors = ['#26A69A' if c >= o else '#EF5350' for c, o in zip(featured_df['Close'], featured_df['Open'])]
    fig_tech.add_trace(go.Bar(
        x=featured_df.index, y=featured_df['Volume'],
        name='Volume', marker_color=colors
    ), row=3, col=1)
    
    fig_tech.update_layout(
        template="plotly_dark", height=700, showlegend=True, xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_tech, use_container_width=True)

# =============================================================================
# TAB 3: BACKTEST RESULTS
# =============================================================================
with tab_backtest:
    st.subheader(f"⚡ Backtest Results — {strategy_choice}")
    
    if enable_garch_risk:
        st.info(f"🛡️ **GARCH Dynamic Risk Control Active**: Dynamic Stop-Loss ({garch_meta.get('latest_daily_vol_pct', 1.5)*garch_multiplier:.1f}%) and volatility-scaled position sizing applied day-by-day.")
        
    raw_m = metrics_res["raw"]
    strat_m = raw_m["Strategy"]
    bm_m = raw_m["Buy & Hold Benchmark"]
    
    m1, m2, m3, m4, m5 = st.columns(5)
    c_ret = "positive-val" if strat_m['Total Return %'] >= 0 else "negative-val"
    c_bm = "positive-val" if bm_m['Total Return %'] >= 0 else "negative-val"
    
    m1.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Strategy Return</div>
        <div class="metric-value {c_ret}">{strat_m['Total Return %']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    m2.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Benchmark Return</div>
        <div class="metric-value {c_bm}">{bm_m['Total Return %']:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    m3.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sharpe Ratio</div>
        <div class="metric-value neutral-val">{strat_m['Sharpe Ratio']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    m4.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Win Rate</div>
        <div class="metric-value neutral-val">{strat_m['Win Rate %']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    m5.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Drawdown</div>
        <div class="metric-value negative-val">{strat_m['Max Drawdown %']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    equity_df = backtest_res["equity_curve"]
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['Strategy Equity'],
        mode='lines', name=f'Strategy ({strategy_choice})',
        line=dict(color='#00E676', width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['Benchmark Equity'],
        mode='lines', name='Buy & Hold Benchmark',
        line=dict(color='#FF9100', width=1.8, dash='dash')
    ))
    fig_equity.update_layout(
        title="Portfolio Equity Curve Over Time (Strategy vs Benchmark)",
        xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Side-by-Side Model Comparison Section (Random Forest / XGBoost vs. LSTM)
    if "AI" in strategy_choice or "LSTM" in strategy_choice:
        st.markdown("---")
        st.markdown("### 🤖 Model Comparison: Random Forest vs. LSTM Sequence Network")
        st.caption("Out-of-sample performance comparison between the primary tabular classifier and the secondary deep learning sequence model.")
        
        with st.spinner("Computing out-of-sample LSTM sequence comparison benchmark..."):
            rf_sig, rf_m_info = train_and_predict_ml_signals(featured_df, horizon=strategy_params.get("horizon", 5), confidence_threshold=strategy_params.get("confidence", 0.55), model_type="Random Forest")
            rf_bt = run_backtest(featured_df, rf_sig, initial_capital=initial_capital, position_size_pct=position_size_pct, transaction_fee_pct=transaction_fee_pct, slippage_pct=slippage_pct, stop_loss_pct=stop_loss_val, use_garch_risk=enable_garch_risk)
            rf_met = calculate_metrics(rf_bt["equity_curve"], rf_bt["trade_log"], initial_capital=initial_capital)
            
            lstm_sig, lstm_m_info = train_and_predict_lstm_signals(featured_df, horizon=strategy_params.get("horizon", 5), confidence_threshold=strategy_params.get("confidence", 0.55), seq_length=strategy_params.get("seq_length", 10))
            lstm_bt = run_backtest(featured_df, lstm_sig, initial_capital=initial_capital, position_size_pct=position_size_pct, transaction_fee_pct=transaction_fee_pct, slippage_pct=slippage_pct, stop_loss_pct=stop_loss_val, use_garch_risk=enable_garch_risk)
            lstm_met = calculate_metrics(lstm_bt["equity_curve"], lstm_bt["trade_log"], initial_capital=initial_capital)
            
            comp_model_table = build_model_comparison_table(rf_m_info, rf_met, lstm_m_info, lstm_met)
            st.table(comp_model_table.set_index("Evaluation Dimension"))
            
    # Trade Markers Chart
    st.markdown("### Trade Executions & Entry/Exit Markers")
    fig_trades = go.Figure()
    fig_trades.add_trace(go.Scatter(
        x=featured_df.index, y=featured_df['Close'],
        mode='lines', name='Close Price', line=dict(color='#90A4AE', width=1.5)
    ))
    
    trade_log_df = backtest_res["trade_log"]
    if not trade_log_df.empty:
        fig_trades.add_trace(go.Scatter(
            x=trade_log_df['Entry Date'], y=trade_log_df['Entry Price'],
            mode='markers', name='Buy Entry', marker=dict(symbol='triangle-up', size=11, color='#00E676')
        ))
        fig_trades.add_trace(go.Scatter(
            x=trade_log_df['Exit Date'], y=trade_log_df['Exit Price'],
            mode='markers', name='Sell Exit', marker=dict(symbol='triangle-down', size=11, color='#FF5252')
        ))
        
    fig_trades.update_layout(
        title="Executed Trades Overlay on Stock Price",
        xaxis_title="Date", yaxis_title="Price (₹)",
        template="plotly_dark", height=400
    )
    st.plotly_chart(fig_trades, use_container_width=True)
    
    # Feature Importance Chart
    if "feature_importances" in strat_meta:
        st.markdown("### 📊 Primary Model Feature Importances")
        importances = strat_meta["feature_importances"]
        feat_df = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values(by="Importance", ascending=True)
        
        fig_feat = go.Figure(go.Bar(
            x=feat_df['Importance'], y=feat_df['Feature'],
            orientation='h', marker_color='#29B6F6'
        ))
        fig_feat.update_layout(
            title="Feature Importance Weights", template="plotly_dark", height=250, margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Detailed Performance Metrics Table")
    st.table(metrics_res["comparison_table"].set_index("Metric"))
    
    # Executed Trade Log with Per-Trade Indicator Rationale
    st.markdown("### Executed Trade Log (with Per-Trade Indicator Rationale)")
    if not trade_log_df.empty:
        formatted_trade_log = trade_log_df.copy()
        formatted_trade_log['Entry Date'] = pd.to_datetime(formatted_trade_log['Entry Date']).dt.strftime('%Y-%m-%d')
        formatted_trade_log['Exit Date'] = pd.to_datetime(formatted_trade_log['Exit Date']).dt.strftime('%Y-%m-%d')
        formatted_trade_log['Entry Price'] = formatted_trade_log['Entry Price'].apply(lambda x: f"₹{x:,.2f}")
        formatted_trade_log['Exit Price'] = formatted_trade_log['Exit Price'].apply(lambda x: f"₹{x:,.2f}")
        formatted_trade_log['Gross PnL'] = formatted_trade_log['Gross PnL'].apply(lambda x: f"₹{x:,.2f}")
        formatted_trade_log['Net PnL'] = formatted_trade_log['Net PnL'].apply(lambda x: f"₹{x:,.2f}")
        formatted_trade_log['Return %'] = formatted_trade_log['Return %'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(formatted_trade_log, use_container_width=True, height=300)
    else:
        # Section 11 Explicit Empty State
        st.info("ℹ️ Zero trades were executed during this backtest window for the selected parameters.")

# =============================================================================
# TAB 4: ROBUSTNESS & VALIDATION (SECTION 10)
# =============================================================================
with tab_robustness:
    st.subheader("🛡️ Model Robustness & Validation Engine")
    st.markdown("Evaluates out-of-sample stability via **Walk-Forward Rolling Validation** and **Multi-Stock Sector Robustness Batch Testing**.")
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        st.markdown("### 🔄 1. Walk-Forward Rolling Validation")
        st.caption("Retrains the model on rolling 250-day windows and evaluates on sequential 60-day out-of-sample test windows.")
        
        if st.button("Run Walk-Forward Validation"):
            with st.spinner("Executing rolling walk-forward validation windows..."):
                wf_signals, wf_meta = run_walk_forward_validation(featured_df)
                wf_bt = run_backtest(featured_df, wf_signals, initial_capital=initial_capital, use_garch_risk=enable_garch_risk)
                wf_met = calculate_metrics(wf_bt["equity_curve"], wf_bt["trade_log"], initial_capital=initial_capital)
                
                st.success(f"✓ Walk-Forward completed over {wf_meta['num_windows']} rolling windows.")
                st.write(f"**Mean Out-of-Sample Accuracy**: `{wf_meta['mean_test_acc']*100:.1f}%`")
                st.write(f"**Walk-Forward Strategy Return**: `{wf_met['raw']['Strategy']['Total Return %']:+.2f}%`")
                
                wf_table = pd.DataFrame(wf_meta['window_results'])
                st.dataframe(wf_table, use_container_width=True, height=200)

    with col_w2:
        st.markdown("### 🌐 2. Multi-Stock Sector Robustness Test")
        st.caption("Executes the strategy across a basket of 10 top NSE equities (Banking, IT, Auto, FMCG, Pharma) to report performance ranges.")
        
        if st.button("Run Multi-Stock Sector Robustness Test"):
            with st.spinner("Running batch simulation across 10 NSE sector stocks..."):
                rob_res = run_multi_stock_robustness_test(strategy_name=strategy_choice, period=selected_period, initial_capital=initial_capital, use_garch_risk=enable_garch_risk)
                
                if "error" in rob_res:
                    st.error(rob_res["error"])
                else:
                    st.success("✓ Sector robustness batch testing complete!")
                    st.markdown("#### Performance Range Summary")
                    st.table(rob_res["range_table"].set_index("Range Statistic"))
                    
                    st.markdown("#### Per-Stock Performance Breakdown")
                    st.dataframe(rob_res["stock_details_df"], use_container_width=True, height=220)

# =============================================================================
# TAB 5: AI FORECAST
# =============================================================================
with tab_forecast:
    st.info("ℹ️ **Notice — Forecast vs. Backtest**: This tab computes a forward linear trend projection. It is completely distinct from the historical trade simulation in the Backtest tab.")
    
    st.subheader("Linear Regression Price Trend Projection")
    forecast_days = st.slider("Forecast Horizon (Days into future):", min_value=5, max_value=60, value=30)
    
    df_fc = featured_df.dropna(subset=['Close']).copy()
    X = np.arange(len(df_fc)).reshape(-1, 1)
    y = df_fc['Close'].values
    
    model_lr = lm.LinearRegression()
    model_lr.fit(X, y)
    
    future_X = np.arange(len(df_fc), len(df_fc) + forecast_days).reshape(-1, 1)
    future_pred = model_lr.predict(future_X)
    
    last_date = df_fc.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    
    y_pred_hist = model_lr.predict(X)
    residual_std = np.std(y - y_pred_hist)
    
    upper_bound = future_pred + (1.96 * residual_std)
    lower_bound = future_pred - (1.96 * residual_std)
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df_fc.index[-100:], y=df_fc['Close'].iloc[-100:],
        mode='lines', name='Historical Close', line=dict(color='#2962FF', width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=future_pred,
        mode='lines', name='Forecast Trend', line=dict(color='#FFD600', width=2, dash='dash')
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=upper_bound,
        mode='lines', name='Upper 95% Bound', line=dict(color='rgba(255, 214, 0, 0.2)', width=0)
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=lower_bound,
        mode='lines', name='Lower 95% Bound', line=dict(color='rgba(255, 214, 0, 0.2)', width=0),
        fill='tonexty', fillcolor='rgba(255, 214, 0, 0.1)'
    ))
    
    fig_fc.update_layout(
        title=f"Linear Regression Forecast for Next {forecast_days} Business Days",
        xaxis_title="Date", yaxis_title="Price (₹)",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# =============================================================================
# TAB 6: DATA & EXPORT
# =============================================================================
with tab_data:
    st.subheader("Raw Data & Export Center")
    
    st.markdown("### Stock OHLCV Historical Data")
    st.dataframe(featured_df, use_container_width=True, height=250)
    
    c_d1, c_d2, c_d3 = st.columns(3)
    with c_d1:
        csv_ohlcv = featured_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Stock Data CSV",
            data=csv_ohlcv,
            file_name=f"{ticker_symbol}_historical_data.csv",
            mime="text/csv"
        )
    with c_d2:
        trade_log_export = backtest_res["trade_log"]
        csv_trades = trade_log_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Trade Log CSV",
            data=csv_trades,
            file_name=f"{ticker_symbol}_{strategy_choice}_trade_log.csv",
            mime="text/csv"
        )
    with c_d3:
        metrics_export = metrics_res["comparison_table"]
        csv_metrics = metrics_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Metrics Summary CSV",
            data=csv_metrics,
            file_name=f"{ticker_symbol}_{strategy_choice}_metrics.csv",
            mime="text/csv"
        )
