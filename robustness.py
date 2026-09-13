import pandas as pd
import numpy as np
from typing import Dict, Any, List
from data_loader import load_stock_data, POPULAR_NSE_STOCKS
from indicators import add_indicators
from volatility_model import fit_garch_volatility
from strategies import generate_signals
from backtester import run_backtest
from metrics import calculate_metrics

ROBUSTNESS_BASKET = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "TATASTEEL.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS"
]

def run_multi_stock_robustness_test(
    strategy_name: str = "AI Hybrid (Random Forest + GARCH)",
    period: str = "1y",
    initial_capital: float = 100000.0,
    use_garch_risk: bool = True
) -> Dict[str, Any]:
    """
    Run strategy across a basket of 10 NSE stocks from different sectors to calculate range statistics
    (Best, Worst, Average, Median) rather than a single isolated backtest result.
    """
    results: List[Dict[str, Any]] = []
    
    for ticker in ROBUSTNESS_BASKET:
        try:
            raw_df, info = load_stock_data(ticker, period=period)
            featured_df = add_indicators(raw_df)
            garch_df, _ = fit_garch_volatility(featured_df)
            
            signals, _ = generate_signals(raw_df, strategy_name, {"horizon": 5, "confidence": 0.55})
            bt_res = run_backtest(garch_df, signals, initial_capital=initial_capital, use_garch_risk=use_garch_risk)
            met_res = calculate_metrics(bt_res["equity_curve"], bt_res["trade_log"], initial_capital=initial_capital)
            
            strat_m = met_res["raw"]["Strategy"]
            bm_m = met_res["raw"]["Buy & Hold Benchmark"]
            
            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Company": POPULAR_NSE_STOCKS.get(ticker, ticker),
                "Strategy Return %": float(strat_m["Total Return %"]),
                "Benchmark Return %": float(bm_m["Total Return %"]),
                "Sharpe Ratio": float(strat_m["Sharpe Ratio"]),
                "Max Drawdown %": float(strat_m["Max Drawdown %"]),
                "Win Rate %": float(strat_m["Win Rate %"]),
                "Trades": int(strat_m["Total Trades"])
            })
        except Exception:
            continue
            
    if not results:
        return {"error": "Multi-stock robustness test failed to load data."}
        
    res_df = pd.DataFrame(results)
    
    returns = res_df["Strategy Return %"]
    sharpes = res_df["Sharpe Ratio"]
    drawdowns = res_df["Max Drawdown %"]
    
    range_summary = {
        "Best Return %": float(returns.max()),
        "Worst Return %": float(returns.min()),
        "Average Return %": float(returns.mean()),
        "Median Return %": float(returns.median()),
        "Mean Sharpe Ratio": float(sharpes.mean()),
        "Mean Max Drawdown %": float(drawdowns.mean()),
        "Total Stocks Tested": len(res_df)
    }
    
    range_table = pd.DataFrame({
        "Range Statistic": [
            "Best Stock Return (%)",
            "Worst Stock Return (%)",
            "Average Return (%) across Sector Basket",
            "Median Return (%) across Sector Basket",
            "Average Sharpe Ratio",
            "Average Max Drawdown (%)",
            "Total Stocks Tested"
        ],
        "Value": [
            f"{range_summary['Best Return %']:+.2f}%",
            f"{range_summary['Worst Return %']:+.2f}%",
            f"{range_summary['Average Return %']:+.2f}%",
            f"{range_summary['Median Return %']:+.2f}%",
            f"{range_summary['Mean Sharpe Ratio']:.2f}",
            f"{range_summary['Mean Max Drawdown %']:.2f}%",
            f"{range_summary['Total Stocks Tested']} stocks"
        ]
    })
    
    return {
        "range_summary": range_summary,
        "range_table": range_table,
        "stock_details_df": res_df
    }
