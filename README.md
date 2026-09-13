# 📈 AI-Powered Trading Strategy Backtester

> Simulate, compare, and validate trading strategies on Indian (NSE) equities — with a Hybrid AI model stack, GARCH volatility layer, walk-forward validation, and zero look-ahead bias.

**Built by:** Mukund Chaurasiya | **Status:** v1.0 — Internship Review Build

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://ai-trading-backtester.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### 🌐 [https://ai-trading-backtester.streamlit.app/](https://ai-trading-backtester.streamlit.app/)

---

## 🎯 What This Project Does

Most free tools show indicators (RSI, MA) without connecting them to profitability, or forecast prices without simulating actual trades. This backtester bridges that gap:

- **Picks any NSE-listed stock**, fetches historical OHLCV data, and runs a full day-by-day simulation
- **Supports 6 strategies** — from simple MA crossovers to a Hybrid AI stack combining Random Forest, XGBoost, GARCH volatility, and a PyTorch LSTM for honest model comparison
- **Applies realistic friction** — configurable transaction costs, slippage, static or GARCH-dynamic stop-loss, and volatility-scaled position sizing
- **Validates rigorously** — walk-forward rolling validation and multi-stock sector robustness testing, reporting results as ranges (Best/Worst/Average), not lucky single numbers
- **Explains its decisions** — feature importance charts, per-trade indicator rationale (RSI, MA Ratio, BB %B at entry), and always shows Random Forest vs. LSTM side-by-side

---

## 📸 Application Tabs

| Tab | What You See |
|---|---|
| **📊 Overview** | Current price, 52W High/Low, GARCH annualised volatility, historical price chart |
| **📈 Technical Analysis** | Candlestick/line chart with MA overlays, GARCH Volatility Bands, Bollinger Bands, RSI, Volume |
| **⚡ Backtest Results** | Equity curve vs Buy & Hold, trade markers, RF vs LSTM model comparison table, feature importance, trade log with per-trade indicator rationale |
| **🛡️ Robustness & Validation** | Walk-forward rolling validation (rolling windows), Multi-stock sector robustness batch test across 10 NSE stocks |
| **🤖 AI Forecast** | Linear regression price trend projection (clearly separated from backtest — no conflation) |
| **📁 Data & Export** | OHLCV data viewer, CSV downloads for stock data, trade log, and metrics summary |

---

## 🏗️ Architecture

```
main.py                 ← Streamlit UI (6 tabs, sidebar controls)
│
├── data_loader.py      ← yfinance OHLCV fetch, forward-fill error handling, caching
├── indicators.py       ← MA, RSI, Bollinger Bands, %B, rolling volatility, volume ratio
├── volatility_model.py ← GARCH(1,1) daily conditional volatility + price bands (arch library)
│
├── strategies.py       ← Unified signal generator for all 6 strategy types
│   ├── ml_model.py     ← Random Forest + XGBoost classifier (tabular, time-safe split)
│   └── lstm_model.py   ← PyTorch LSTM sequence model (comparison benchmark)
│
├── backtester.py       ← Day-by-day simulation: cash, shares, fees, slippage, GARCH risk, trade log
├── metrics.py          ← Sharpe, CAGR, Max Drawdown, Win Rate, RF vs LSTM comparison table
│
├── walk_forward.py     ← Rolling walk-forward validation engine
└── robustness.py       ← Multi-stock sector robustness batch runner
```

### Data Flow

```
User selects stock + strategy + risk parameters
        ↓
data_loader.py → fetches OHLCV (yfinance), forward-fills gaps
        ↓
indicators.py → MA, RSI, Bollinger Bands, returns, volatility
        ↓
volatility_model.py → GARCH(1,1) fits conditional volatility bands
        ↓
strategies.py → generates Buy / Hold / Sell signals
   (Rule-based  OR  ml_model.py RF/XGBoost  OR  lstm_model.py LSTM)
        ↓
backtester.py → simulates trades day-by-day with GARCH dynamic risk
        ↓
metrics.py → computes performance stats + model comparison table
        ↓
main.py → renders all charts, tables, and export buttons
```

---

## 🤖 AI / ML Strategy Stack

### Primary — Random Forest / XGBoost Classifier
- Trains on engineered tabular features: `RSI`, `MA Ratio`, `BB %B`, `Daily Return`, `Rolling Volatility`, `Volume Ratio`
- Predicts N-day price direction (binary classification)
- Strict **chronological 70/30 train/test split** — zero look-ahead bias
- Outputs feature importance chart so decisions are explainable

### Volatility Layer — GARCH(1,1)
- Fits a GARCH(1,1) model on log returns via the `arch` library
- Produces daily conditional volatility forecasts $\sigma_t$
- Used to:
  - Overlay **GARCH Volatility Price Bands** ($P_t \pm k \cdot \sigma_t$) on the price chart (distinct from Bollinger Bands)
  - Apply **dynamic stop-loss** thresholds ($k \times \sigma_t$ per day)
  - Apply **volatility-inverse position sizing** (scale down trades when market is more volatile)

### Comparison Benchmark — PyTorch LSTM Sequence Model
- Processes sliding 10-day windows of feature sequences
- Trained on the same 70% chronological training window
- Evaluated strictly **out-of-sample** — never cherry-picked
- Shown **side-by-side with Random Forest** in the model comparison table

### Walk-Forward Validation
- Retrains the model on rolling 250-day windows, tests on the next 60-day out-of-sample window
- Chains out-of-sample predictions end-to-end to build a realistic live-execution equity curve
- Tracks **feature importance drift** across rolling windows as an overfitting check

---

## 📦 Installation

```bash
git clone https://github.com/mukundpy/AI-Backtester.git
cd AI-Backtester
pip install -r requirements.txt
```

### Requirements
```
streamlit>=1.25.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
arch>=6.0.0
xgboost>=1.7.0
torch>=2.0.0
```

---

## 🚀 Run the App

```bash
streamlit run main.py
```

Then open **http://localhost:8501** in your browser.

---

## 📐 Available Strategies

| Strategy | Type | Description |
|---|---|---|
| **MA Crossover** | Rule-Based | Golden/Death Cross — Fast MA vs Slow MA |
| **RSI Threshold** | Rule-Based | Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought) |
| **Bollinger Reversion** | Rule-Based | Buy below lower band, Sell above upper band |
| **AI Hybrid (Random Forest + GARCH)** | ML + Volatility | RF classifier with GARCH dynamic risk management |
| **AI Hybrid (XGBoost + GARCH)** | ML + Volatility | XGBoost classifier with GARCH dynamic risk management |
| **LSTM Sequence Model** | Deep Learning | PyTorch LSTM on 10-day feature windows (comparison benchmark) |

---

## 📊 Performance Metrics Computed

| Metric | Description |
|---|---|
| **Total Return %** | Portfolio growth from start to end |
| **CAGR %** | Compounded Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return (annualised, 5% risk-free rate) |
| **Max Drawdown %** | Largest peak-to-trough portfolio loss |
| **Max Drawdown Duration** | How many days the portfolio stayed underwater |
| **Win Rate %** | Percentage of profitable closed trades |
| **Profit Factor** | Total gains ÷ Total losses |
| **Total Trades** | Number of round-trip trades executed |
| **Avg Trade P&L** | Average net profit per trade |

All metrics are shown **side-by-side vs Buy & Hold benchmark** — never just the strategy alone.

---

## 🛡️ Key Safeguards

| Concern | How It's Handled |
|---|---|
| **Look-ahead bias** | Strictly chronological train/test splits; features at time *t* never use data from *t+1* |
| **Overfitting** | Walk-forward validation + in-sample vs out-of-sample accuracy comparison |
| **Survivorship bias** | Multi-stock sector basket testing reports Best/Worst/Average ranges |
| **Unrealistic results** | Transaction costs (0.1%) and slippage (0.05%) applied on every simulated trade |
| **Single lucky backtest** | Walk-forward rolling validation + robustness batch across 10 NSE equities |

---

## ⚠️ Error Handling

| Scenario | Behaviour |
|---|---|
| Stock data < 50 bars | ML strategies automatically disabled; rule-based only |
| NaN / trading halt gaps | Auto forward-filled (`ffill()`), visible note shown to user |
| yfinance API failure | "Retry Data Fetch" button shown instead of silent crash |
| Zero trades generated | Explicit empty state info card shown instead of blank chart |

---

## 📂 Project Structure

```
AI-Backtester/
├── main.py              # Streamlit UI orchestration
├── data_loader.py       # yfinance data fetch + error handling
├── indicators.py        # Technical indicators
├── volatility_model.py  # GARCH(1,1) volatility model
├── ml_model.py          # Random Forest + XGBoost classifier
├── lstm_model.py        # PyTorch LSTM sequence model
├── strategies.py        # Signal generators (rule-based + AI)
├── backtester.py        # Simulation engine + per-trade rationale
├── metrics.py           # Performance metrics + model comparison
├── walk_forward.py      # Walk-forward rolling validation
├── robustness.py        # Multi-stock sector robustness testing
├── requirements.txt     # Python dependencies
├── AI_Backtester_PRD.md # Full Product Requirements Document
└── .gitignore
```

---

## ⚠️ Disclaimer

This project is built for **educational and internship portfolio purposes only**. It is not financial advice, and backtested performance does not guarantee future returns. Never use this to make real investment decisions.

---

## 📄 License

MIT License — free to use, fork, and build upon.
