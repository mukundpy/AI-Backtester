import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def calculate_max_drawdown(equity_series: pd.Series) -> Tuple[float, int]:
    """Calculate maximum drawdown percentage and maximum drawdown duration in days."""
    if equity_series.empty or len(equity_series) < 2:
        return 0.0, 0
        
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = float(drawdown.min() * 100.0) # negative value
    
    is_drawdown = drawdown < 0
    dd_durations = []
    current_duration = 0
    
    for in_dd in is_drawdown:
        if in_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                dd_durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        dd_durations.append(current_duration)
        
    max_dd_duration = max(dd_durations) if dd_durations else 0
    return max_dd, max_dd_duration

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe Ratio (assuming 252 trading days/year)."""
    clean_returns = daily_returns.dropna()
    if len(clean_returns) < 2 or clean_returns.std() == 0:
        return 0.0
        
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = clean_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return float(sharpe)

def calculate_metrics(
    equity_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.05
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for both Strategy and Benchmark.
    """
    strat_equity = equity_df['Strategy Equity']
    bm_equity = equity_df['Benchmark Equity']
    
    total_days = max(1, (equity_df.index[-1] - equity_df.index[0]).days)
    years = total_days / 365.25
    
    strat_final = strat_equity.iloc[-1]
    strat_tot_ret = ((strat_final / initial_capital) - 1.0) * 100.0
    strat_cagr = (((strat_final / initial_capital) ** (1.0 / max(years, 0.08))) - 1.0) * 100.0 if years > 0 else strat_tot_ret
    strat_daily_ret = strat_equity.pct_change()
    strat_sharpe = calculate_sharpe_ratio(strat_daily_ret, risk_free_rate)
    strat_max_dd, strat_dd_days = calculate_max_drawdown(strat_equity)
    
    bm_final = bm_equity.iloc[-1]
    bm_tot_ret = ((bm_final / initial_capital) - 1.0) * 100.0
    bm_cagr = (((bm_final / initial_capital) ** (1.0 / max(years, 0.08))) - 1.0) * 100.0 if years > 0 else bm_tot_ret
    bm_daily_ret = bm_equity.pct_change()
    bm_sharpe = calculate_sharpe_ratio(bm_daily_ret, risk_free_rate)
    bm_max_dd, bm_dd_days = calculate_max_drawdown(bm_equity)
    
    total_trades = len(trade_log)
    if total_trades > 0:
        winning_trades = trade_log[trade_log['Net PnL'] > 0]
        losing_trades = trade_log[trade_log['Net PnL'] < 0]
        win_rate = (len(winning_trades) / total_trades) * 100.0
        total_gains = winning_trades['Net PnL'].sum()
        total_losses = abs(losing_trades['Net PnL'].sum())
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (total_gains if total_gains > 0 else 0.0)
        avg_trade_pnl = trade_log['Net PnL'].mean()
        avg_holding_days = trade_log['Holding Days'].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_holding_days = 0.0
        
    metrics_summary = {
        "Strategy": {
            "Total Return %": strat_tot_ret,
            "CAGR (Annual Return) %": strat_cagr,
            "Sharpe Ratio": strat_sharpe,
            "Max Drawdown %": strat_max_dd,
            "Max Drawdown Duration (days)": strat_dd_days,
            "Win Rate %": win_rate,
            "Total Trades": total_trades,
            "Profit Factor": profit_factor,
            "Avg Trade P&L (₹)": avg_trade_pnl,
            "Avg Holding Period (days)": avg_holding_days
        },
        "Buy & Hold Benchmark": {
            "Total Return %": bm_tot_ret,
            "CAGR (Annual Return) %": bm_cagr,
            "Sharpe Ratio": bm_sharpe,
            "Max Drawdown %": bm_max_dd,
            "Max Drawdown Duration (days)": bm_dd_days,
            "Win Rate %": "N/A",
            "Total Trades": 1,
            "Profit Factor": "N/A",
            "Avg Trade P&L (₹)": "N/A",
            "Avg Holding Period (days)": total_days
        }
    }
    
    comparison_df = pd.DataFrame({
        "Metric": [
            "Total Return (%)",
            "Annualized Return / CAGR (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Max Drawdown Duration (days)",
            "Win Rate (%)",
            "Total Trades",
            "Profit Factor",
            "Avg Trade P&L (₹)"
        ],
        "Strategy": [
            f"{strat_tot_ret:+.2f}%",
            f"{strat_cagr:+.2f}%",
            f"{strat_sharpe:.2f}",
            f"{strat_max_dd:.2f}%",
            f"{strat_dd_days} days",
            f"{win_rate:.1f}%",
            f"{total_trades}",
            f"{profit_factor:.2f}",
            f"₹{avg_trade_pnl:,.2f}"
        ],
        "Buy & Hold Benchmark": [
            f"{bm_tot_ret:+.2f}%",
            f"{bm_cagr:+.2f}%",
            f"{bm_sharpe:.2f}",
            f"{bm_max_dd:.2f}%",
            f"{bm_dd_days} days",
            "N/A",
            "1",
            "N/A",
            "N/A"
        ]
    })
    
    return {
        "raw": metrics_summary,
        "comparison_table": comparison_df
    }

def build_model_comparison_table(
    rf_meta: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    lstm_meta: Dict[str, Any],
    lstm_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Build side-by-side model benchmark comparison table:
    Primary Model (Random Forest / XGBoost) vs Secondary Model (LSTM Sequence Network).
    """
    rf_strat_m = rf_metrics["raw"]["Strategy"]
    lstm_strat_m = lstm_metrics["raw"]["Strategy"]
    
    comp_df = pd.DataFrame({
        "Evaluation Dimension": [
            "Model Architecture",
            "Out-of-Sample Accuracy (%)",
            "In-Sample Training Accuracy (%)",
            "Chronological Split Date",
            "Strategy Total Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Total Executed Trades"
        ],
        "Primary Model (Random Forest / XGBoost)": [
            rf_meta.get("model_type", "Random Forest Classifier"),
            f"{rf_meta.get('test_acc', 0.0)*100:.1f}%",
            f"{rf_meta.get('train_acc', 0.0)*100:.1f}%",
            rf_meta.get("split_date", "N/A"),
            f"{rf_strat_m['Total Return %']:+.2f}%",
            f"{rf_strat_m['Sharpe Ratio']:.2f}",
            f"{rf_strat_m['Max Drawdown %']:.2f}%",
            f"{rf_strat_m['Win Rate %']:.1f}%",
            f"{rf_strat_m['Total Trades']}"
        ],
        "Secondary Comparison Model (LSTM)": [
            lstm_meta.get("model_type", "PyTorch LSTM Sequence Net"),
            f"{lstm_meta.get('test_acc', 0.0)*100:.1f}%",
            f"{lstm_meta.get('train_acc', 0.0)*100:.1f}%",
            lstm_meta.get("split_date", "N/A"),
            f"{lstm_strat_m['Total Return %']:+.2f}%",
            f"{lstm_strat_m['Sharpe Ratio']:.2f}",
            f"{lstm_strat_m['Max Drawdown %']:.2f}%",
            f"{lstm_strat_m['Win Rate %']:.1f}%",
            f"{lstm_strat_m['Total Trades']}"
        ]
    })
    
    return comp_df
