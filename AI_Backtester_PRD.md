# Product Requirements Document (PRD)
## AI-Powered Trading Strategy Backtester

**Author:** Mukund Chaurasiya
**Version:** 1.0
**Date:** September 2026
**Status:** Draft — for internship review

---

## 1. Overview

### 1.1 Problem Statement
Retail and beginner investors, and even student developers exploring quant finance, lack an easy way to test whether a trading idea (technical or AI-driven) would have actually worked historically — before risking real capital. Most free tools either show raw indicators (RSI, MA) without connecting them to profitability, or offer forecasting without simulating actual trades.

### 1.2 Product Vision
Build a web-based application that lets a user select an Indian stock, define or choose a trading strategy (rule-based or ML-driven), simulate that strategy over historical data, and see clear, quantified performance results compared to a buy-and-hold baseline.

### 1.3 Goals
- Turn the existing forecasting/technical-analysis app (Indian Stock Forecast Pro) into a genuine **backtesting engine**.
- Support both simple rule-based strategies and an AI/ML-based signal generator.
- Present results in a way that's understandable to a non-technical reviewer, but rigorous enough to satisfy a technical one.

### 1.4 Non-Goals (v1)
- Live/paper trading execution (no real broker integration)
- Options, futures, or derivatives — equities only
- Multi-asset portfolio optimization (single stock per backtest run in v1)
- Real-time streaming data (daily OHLCV is sufficient for v1)

---

## 2. Target Users

| User | Need |
|---|---|
| Internship reviewer / manager | Wants to see rigor: correct methodology, clear metrics, no look-ahead bias |
| Student/beginner investor | Wants to understand "would this strategy have worked?" in plain language |
| Mukund (builder) | Wants a strong, demonstrable AI/ML + finance portfolio project |

---

## 3. User Stories

1. As a user, I want to pick any NSE-listed stock and a date range, so I can test a strategy on data relevant to me.
2. As a user, I want to choose between a rule-based strategy (MA crossover, RSI) and an AI-based strategy (ML classifier), so I can compare approaches.
3. As a user, I want to see my strategy's total return, Sharpe ratio, max drawdown, and win rate, so I can judge if it's actually good.
4. As a user, I want to see my strategy's performance plotted against simple buy-and-hold, so I know if the extra complexity was worth it.
5. As a user, I want to download my backtest results as CSV, so I can share or analyze them further.
6. As a user, I want to configure position sizing, stop-loss, and transaction costs, so results are realistic.

---

## 4. Functional Requirements & Features

### 4.1 Data Layer
- Pull historical NSE stock data via `yfinance` (existing capability).
- Support configurable time period: 1mo to 5y (existing).
- Cache data per session to avoid repeated API calls (existing via `@st.cache_data`).

### 4.2 Feature Engineering (existing, to extend)
- Moving Averages (20/50) — existing
- RSI (14) — existing
- Bollinger Bands — existing
- **New:** Volume-based features, daily returns, rolling volatility — needed as ML model inputs

### 4.3 Strategy Engine (new — core deliverable)
- **Rule-Based Strategies (v1):**
  - MA Crossover (Golden Cross / Death Cross)
  - RSI Threshold (buy < 30, sell > 70)
  - Bollinger Band Reversion
- **AI-Based Strategy (v1) — Hybrid Model Stack:**
  - **Primary signal — Random Forest / XGBoost classifier:** predicts next N-day price direction using engineered features (RSI, MA, returns, volume). Chosen as the primary model for its strong accuracy-to-effort ratio on tabular financial data and built-in feature importance, which makes it easy to explain and defend in review.
  - **Volatility layer — GARCH(1,1):** forecasts near-term volatility (not price) from historical returns, using the `arch` library. Used to make position sizing and stop-loss thresholds dynamic and risk-aware, rather than fixed percentages.
  - **Secondary/comparison model — LSTM:** a sequence model trained on the same feature set, run alongside the Random Forest as a benchmark. Included to demonstrate awareness of deep learning approaches and to produce an honest "simpler vs. more complex model" comparison — LSTMs are prone to overfitting on noisy financial data, so results are reported out-of-sample only, never cherry-picked.
  - All models use strictly time-respecting train/test splits — no look-ahead bias.
  - Classifier probability output converted into buy/hold/sell signal via configurable threshold.

### 4.4 Backtesting Simulation Engine (new — core deliverable)
- Day-by-day simulation loop:
  - Track cash, shares held, portfolio value at each timestep
  - Execute buy/sell based on active strategy's signal
  - Apply configurable transaction cost (%) and slippage
  - Apply configurable position sizing (% of capital per trade) and optional stop-loss
- Output: full equity curve (portfolio value over time) + trade log (entry/exit price, date, P&L per trade)

### 4.5 Performance Metrics (new)
- Total Return %
- Annualized Return %
- Sharpe Ratio
- Max Drawdown %
- Win Rate % (profitable trades / total trades)
- Number of Trades
- Benchmark comparison: same metrics for Buy & Hold over identical period

### 4.6 Visualization (extend existing)
- Existing: price chart (line/candlestick), RSI chart, volume chart, forecast chart
- **New:** Equity curve chart — Strategy vs. Buy & Hold, overlaid
- **New:** Trade markers on price chart (green triangle = buy, red triangle = sell)
- **New:** Metrics comparison table (Strategy vs. Benchmark side-by-side)
- **New:** Model comparison table — Random Forest vs. LSTM out-of-sample accuracy and resulting backtest performance, shown side-by-side
- **New:** GARCH-forecasted volatility band overlaid on the price chart, distinct from the existing Bollinger Bands

### 4.7 Export & Reporting (extend existing)
- Existing: download raw OHLCV data as CSV
- **New:** download trade log as CSV
- **New:** download summary metrics as CSV or a formatted text report

---

## 5. UI / UX Design

### 5.1 Layout (Streamlit, extending current structure)

**Sidebar (left panel) — Configuration:**
1. Stock search & select (existing)
2. Time period selector (existing)
3. **New:** Strategy selector — dropdown: `["MA Crossover", "RSI Threshold", "Bollinger Reversion", "AI Model (ML Classifier)"]`
4. **New:** Strategy parameters (dynamic based on selection):
   - MA Crossover → fast/slow window sliders
   - RSI → oversold/overbought threshold sliders
   - AI Model → prediction horizon slider (1–10 days), confidence threshold slider
5. **New:** Risk settings — position size % slider, stop-loss % toggle+slider, transaction cost % input
6. Chart type toggle (existing)

**Main panel — Tabs (new structure to organize growing content):**

| Tab | Content |
|---|---|
| **Overview** | Key metrics cards (current price, daily change, 52w high/low) — existing |
| **Technical Analysis** | Price chart with MA/Bollinger overlays, RSI chart — existing |
| **Backtest Results** *(new)* | Equity curve (strategy vs buy & hold), metrics comparison table, trade log table, trade markers on price chart |
| **AI Forecast** | Existing linear regression forecast, clearly labeled as "price forecast" (distinct from "backtest," to avoid confusing the two) |
| **Data & Export** | Raw data table, CSV downloads — existing, extended with trade log export |

### 5.2 Key UI Principles
- Clearly separate **"Forecast"** (what might happen) from **"Backtest"** (what would have happened) — these were conflated in v1 and this PRD's biggest UX fix.
- Every metric shown must have a plain-language one-line explainer (tooltip or caption), since audience includes non-technical reviewers.
- Strategy vs. Benchmark comparison must always be visible side-by-side, never just the strategy alone — prevents misleadingly good-looking results.
- Retain the friendly, approachable visual style and disclaimers from v1 (educational-use framing, emoji section headers, Indian Rupee formatting).

### 5.3 Wireframe (text description)

```
┌─────────────────────────────────────────────────────────┐
│  📈 AI Trading Strategy Backtester                        │
├───────────────┬───────────────────────────────────────────┤
│  SIDEBAR       │  [Overview] [Technical] [Backtest] [Forecast] [Data]
│                │                                           │
│ Stock: [____]  │  ── Backtest Results Tab ──               │
│ Period: [___]  │                                           │
│ Strategy:      │   Equity Curve Chart                      │
│  [AI Model ▾]  │   ───────────────────                     │
│                │   ╱‾‾‾╲___╱‾‾‾ Strategy                   │
│ Horizon: [5d]  │   ___╱‾‾‾‾‾‾‾‾‾ Buy & Hold                │
│ Confidence:    │                                           │
│  [====----]    │   ┌─────────────┬────────────┐            │
│                │   │ Metric      │ Strat | B&H │            │
│ Position %:    │   ├─────────────┼────────────┤            │
│  [====------]  │   │ Total Return│ +18%  | +11%│            │
│ Stop-loss:     │   │ Sharpe      │ 1.4   | 0.9 │            │
│  ☑ [==-----]   │   │ Max Drawdn  │ -8%   | -15%│            │
│ Txn Cost %:    │   │ Win Rate    │ 62%   | —   │            │
│  [0.1]         │   └─────────────┴────────────┘            │
│                │                                           │
│ [Run Backtest] │   Trade Log Table (scrollable)            │
│                │   [Download Trade Log CSV]                │
└───────────────┴───────────────────────────────────────────┘
```

---

## 6. Technical Architecture

### 6.1 Stack
| Layer | Technology |
|---|---|
| Frontend/UI | Streamlit |
| Data | yfinance (NSE tickers via `.NS` suffix) |
| Feature Engineering | pandas, NumPy |
| Volatility Forecasting | `arch` (GARCH models) |
| ML Models | scikit-learn / XGBoost (Random Forest — primary signal), TensorFlow/Keras or PyTorch (LSTM — secondary/comparison model) |
| Backtesting Engine | Custom Python simulation loop (pandas-based) |
| Visualization | Plotly |
| Deployment | Streamlit Community Cloud |

### 6.2 Proposed Module Structure
```
main.py                 # Streamlit UI orchestration (existing, extended)
data_loader.py          # yfinance fetch + caching (extracted from main.py)
indicators.py           # MA, RSI, Bollinger Bands calculations (extracted)
strategies.py           # NEW: rule-based + ML-based signal generators
backtester.py           # NEW: simulation loop, portfolio tracking
metrics.py              # NEW: Sharpe, drawdown, win rate calculations
ml_model.py              # NEW: Random Forest classifier (primary), time-respecting train/test split
volatility.py            # NEW: GARCH(1,1) volatility forecasting
lstm_model.py            # NEW: LSTM comparison model (secondary/experimental)
requirements.txt
```

### 6.3 Data Flow
```
User selects stock + strategy + params
        ↓
data_loader.py fetches OHLCV
        ↓
indicators.py computes features
        ↓
strategies.py generates buy/sell/hold signals
   (rule-based OR ml_model.py prediction)
        ↓
backtester.py simulates trades day-by-day
        ↓
metrics.py computes performance stats
        ↓
main.py renders charts + tables in UI
```

### 6.4 Key Technical Safeguards
- **No look-ahead bias:** ML model train/test split strictly respects chronological order; features at time *t* never use data from *t+1* onward.
- **Realistic simulation:** transaction costs and slippage applied on every simulated trade.
- **Reproducibility:** random seed fixed for ML models; all strategy parameters logged with each backtest run.

---

## 7. Success Metrics (for this project/internship)

- Backtest engine produces correct, verifiable portfolio value series (validated against a manual hand-calculated example).
- At least 3 rule-based strategies + 1 ML-based strategy implemented and working.
- UI clearly distinguishes forecast vs backtest (no user confusion in a walkthrough demo).
- App runs end-to-end without errors on at least 10 different NSE stocks and multiple time periods.
- Manager/reviewer can articulate the strategy's performance vs. benchmark within 30 seconds of viewing the Backtest tab.

---

## 8. Milestones

| Phase | Deliverable | Est. Time |
|---|---|---|
| 1 | Refactor existing code into modules (data_loader, indicators) | 1–2 days |
| 2 | Build rule-based strategy engine + backtesting simulation loop | 3–4 days |
| 3 | Add performance metrics + equity curve visualization | 2 days |
| 4 | Build hybrid AI signal: Random Forest classifier (primary) + GARCH volatility layer + LSTM (comparison model), all with time-safe validation | 5–6 days |
| 5 | UI overhaul — tabbed layout, strategy config panel, comparison table | 2–3 days |
| 6 | Robustness testing — walk-forward validation, multi-stock/multi-period runs, overfitting checks | 3 days |
| 7 | Error handling & edge cases (short history, missing data, API failures, invalid inputs) | 1–2 days |
| 8 | Explainability additions — feature importance chart, per-trade rationale | 1 day |
| 9 | Testing across multiple stocks/periods, bug fixes, polish | 2 days |
| 10 | Documentation + demo prep | 1 day |

---

## 9. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| ML model may overfit to historical data | Use walk-forward validation; report out-of-sample performance only |
| Results may look artificially good without realistic costs | Enforce transaction costs/slippage as non-optional in simulation |
| Single-stock backtests may not generalize | Test strategy across multiple stocks/sectors before drawing conclusions |
| Scope creep (portfolio-level, live trading) | Explicitly out of scope for v1, documented above |

**Open questions for manager/stakeholder:**
1. Should the ML model support multiple stocks simultaneously, or is single-stock backtesting sufficient for v1?
2. Is there a preferred benchmark besides Buy & Hold (e.g., Nifty 50 index)?
3. What level of explainability is expected for the AI model's decisions (e.g., feature importance)?

---

## 10. Robustness & Validation Strategy

A single backtest on one stock over one period proves very little — it's easy to accidentally cherry-pick a result that looks good by luck. This section ensures the results are trustworthy.

### 10.1 Walk-Forward Validation
Instead of one fixed train/test split, retrain the model periodically as if running it live:
- Split full history into rolling windows (e.g., train on 2 years, test on next 3 months, then slide forward).
- Retrain the Random Forest / LSTM at each step using only data available up to that point.
- Aggregate performance across all windows — this is far more representative of real-world behavior than a single split.

### 10.2 Multi-Stock, Multi-Period Testing
- Run every strategy across a fixed basket of at least 10–15 stocks spanning different sectors (banking, IT, pharma, etc.) and at least 2–3 distinct time periods (including a volatile period, e.g., 2020, and a calmer one).
- Report performance as a **distribution** (average, best case, worst case) rather than a single number — a strategy that only wins on one stock isn't a real strategy.

### 10.3 Overfitting Checks
- Compare in-sample vs. out-of-sample performance for every model; a large gap signals overfitting.
- Track and report the Random Forest's feature importances across different training windows — if importance rankings shift wildly, the model is unstable.

---

## 11. Error Handling & Edge Cases

Needed for the app to be demo-safe and production-credible, not just a happy-path prototype.

| Scenario | Handling |
|---|---|
| Stock has < 60 days of data in selected period | Show a clear warning; disable ML strategy option (needs minimum history), allow rule-based only |
| Missing/NaN values in OHLCV data (e.g., trading halts) | Forward-fill or drop with a visible note on how many rows were affected |
| yfinance API failure / rate limit | Catch exception (already partially handled in `load_stock_data`), show retry option instead of silent failure |
| User selects 0% position size or invalid stop-loss | Validate inputs before running backtest; block "Run Backtest" with inline error message |
| Model prediction confidence never crosses threshold (no trades generated) | Explicitly show "0 trades executed" state rather than a blank/broken chart |
| Extremely short forecast/backtest window (e.g., 1mo) | Warn that results are statistically unreliable at this length |

---

## 12. Explainability

Since this is a finance-adjacent AI tool, being able to explain *why* the model made a decision matters as much as the decision itself — both for your manager's review and for user trust.

- **Feature importance chart:** Show which inputs (RSI, MA, volume, volatility) drove the Random Forest's decisions most, using scikit-learn's built-in `feature_importances_`.
- **Per-trade rationale:** For each trade in the log, optionally show the feature values at that point in time (e.g., "RSI was 28, MA20 crossed above MA50") so a reviewer can sanity-check individual decisions.
- **Model comparison transparency:** When showing Random Forest vs. LSTM results, always present them side-by-side with the same metrics — never present the better-performing model alone, since that would misrepresent the LSTM's actual (likely weaker, more overfit-prone) reliability.

---

## 13. Appendix: Current State (v0 — already built)

The existing "Indian Stock Forecast Pro" app already provides:
- NSE stock search & selection (100+ companies)
- Historical data fetch with configurable period
- Technical indicators: MA20/50, RSI, Bollinger Bands
- Line & candlestick charts
- Simple Linear Regression price forecast with confidence bands
- Returns/volatility/volume analysis
- CSV data export

This PRD builds directly on top of this foundation rather than replacing it.
