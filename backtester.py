import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    position_size_pct: float = 1.0,
    transaction_fee_pct: float = 0.001,  # 0.1% per trade
    slippage_pct: float = 0.0005,        # 0.05% slippage
    stop_loss_pct: float = None,          # Static stop loss e.g. 0.05
    use_garch_risk: bool = False          # Enable GARCH dynamic risk management
) -> Dict[str, Any]:
    """
    Run day-by-day event-driven backtest simulation loop recording per-trade indicator rationale.
    """
    data = df.copy()
    data['Signal'] = signals.reindex(data.index).fillna(0)
    
    cash = float(initial_capital)
    shares = 0
    position_entry_price = 0.0
    position_entry_date = None
    entry_rationale = {}
    
    equity_curve = []
    trades: List[Dict[str, Any]] = []
    
    first_price = data['Close'].iloc[0] * (1.0 + slippage_pct)
    bm_shares = int((initial_capital * (1.0 - transaction_fee_pct)) / first_price) if first_price > 0 else 0
    bm_cash = initial_capital - (bm_shares * first_price * (1.0 + transaction_fee_pct))
    
    for date, row in data.iterrows():
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        sig = row['Signal']
        
        if use_garch_risk and 'Dynamic_Stop_Loss_Pct' in row:
            eff_stop_loss = float(row['Dynamic_Stop_Loss_Pct'])
        else:
            eff_stop_loss = stop_loss_pct
            
        # 1. Stop-Loss Execution check
        if shares > 0 and eff_stop_loss is not None and eff_stop_loss > 0:
            stop_price = position_entry_price * (1.0 - eff_stop_loss)
            if low_price <= stop_price:
                exit_price = max(low_price, stop_price) * (1.0 - slippage_pct)
                gross_value = shares * exit_price
                fee = gross_value * transaction_fee_pct
                net_value = gross_value - fee
                cash += net_value
                
                pnl = net_value - (shares * position_entry_price)
                pnl_pct = (exit_price / position_entry_price - 1.0) * 100.0
                holding_days = (date - position_entry_date).days if hasattr(date - position_entry_date, 'days') else 1
                
                trade_record = {
                    "Entry Date": position_entry_date,
                    "Exit Date": date,
                    "Type": "Long",
                    "Entry Price": position_entry_price,
                    "Exit Price": exit_price,
                    "Shares": shares,
                    "Gross PnL": gross_value - (shares * position_entry_price),
                    "Net PnL": pnl,
                    "Return %": pnl_pct,
                    "Holding Days": max(1, holding_days),
                    "Exit Reason": f"GARCH Stop Loss ({eff_stop_loss*100:.1f}%)" if use_garch_risk else "Stop Loss"
                }
                trade_record.update(entry_rationale)
                trades.append(trade_record)
                
                shares = 0
                position_entry_price = 0.0
                position_entry_date = None
                entry_rationale = {}
                
        # 2. Strategy Signal Execution
        if shares == 0 and sig == 1:
            if use_garch_risk and 'GARCH_Vol_Daily' in row and row['GARCH_Vol_Daily'] > 0:
                vol_scale = min(1.0, 0.015 / row['GARCH_Vol_Daily'])
                effective_pos_pct = position_size_pct * vol_scale
            else:
                effective_pos_pct = position_size_pct
                
            alloc_cash = cash * effective_pos_pct
            buy_price = close_price * (1.0 + slippage_pct)
            cost_per_share = buy_price * (1.0 + transaction_fee_pct)
            
            n_shares = int(alloc_cash / cost_per_share) if cost_per_share > 0 else 0
            if n_shares > 0:
                total_cost = n_shares * cost_per_share
                cash -= total_cost
                shares = n_shares
                position_entry_price = buy_price
                position_entry_date = date
                
                # Record exact indicator rationale at entry
                entry_rationale = {
                    "RSI @ Entry": round(row.get('RSI', 50.0), 1),
                    "MA Ratio @ Entry": round(row.get('MA_Ratio', 1.0), 3),
                    "BB %B @ Entry": round(row.get('BB_PctB', 0.5), 2),
                    "Vol % @ Entry": round(row.get('GARCH_Vol_Daily', 0.015) * 100.0, 2)
                }
                
        elif shares > 0 and sig == -1:
            sell_price = close_price * (1.0 - slippage_pct)
            gross_value = shares * sell_price
            fee = gross_value * transaction_fee_pct
            net_value = gross_value - fee
            cash += net_value
            
            pnl = net_value - (shares * position_entry_price)
            pnl_pct = (sell_price / position_entry_price - 1.0) * 100.0
            holding_days = (date - position_entry_date).days if hasattr(date - position_entry_date, 'days') else 1
            
            trade_record = {
                "Entry Date": position_entry_date,
                "Exit Date": date,
                "Type": "Long",
                "Entry Price": position_entry_price,
                "Exit Price": sell_price,
                "Shares": shares,
                "Gross PnL": gross_value - (shares * position_entry_price),
                "Net PnL": pnl,
                "Return %": pnl_pct,
                "Holding Days": max(1, holding_days),
                "Exit Reason": "Signal Exit"
            }
            trade_record.update(entry_rationale)
            trades.append(trade_record)
            
            shares = 0
            position_entry_price = 0.0
            position_entry_date = None
            entry_rationale = {}
            
        strat_equity = cash + (shares * close_price)
        bm_equity = bm_cash + (bm_shares * close_price)
        
        equity_curve.append({
            "Date": date,
            "Strategy Equity": strat_equity,
            "Benchmark Equity": bm_equity,
            "Strategy Position": 1 if shares > 0 else 0,
            "Close": close_price
        })
        
    if shares > 0 and position_entry_date is not None:
        last_date = data.index[-1]
        last_close = data['Close'].iloc[-1] * (1.0 - slippage_pct)
        gross_value = shares * last_close
        fee = gross_value * transaction_fee_pct
        net_value = gross_value - fee
        
        pnl = net_value - (shares * position_entry_price)
        pnl_pct = (last_close / position_entry_price - 1.0) * 100.0
        holding_days = (last_date - position_entry_date).days if hasattr(last_date - position_entry_date, 'days') else 1
        
        trade_record = {
            "Entry Date": position_entry_date,
            "Exit Date": last_date,
            "Type": "Long",
            "Entry Price": position_entry_price,
            "Exit Price": last_close,
            "Shares": shares,
            "Gross PnL": gross_value - (shares * position_entry_price),
            "Net PnL": pnl,
            "Return %": pnl_pct,
            "Holding Days": max(1, holding_days),
            "Exit Reason": "End of Period"
        }
        trade_record.update(entry_rationale)
        trades.append(trade_record)
        
    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "Entry Date", "Exit Date", "Type", "Entry Price", "Exit Price",
        "Shares", "Gross PnL", "Net PnL", "Return %", "Holding Days", "Exit Reason",
        "RSI @ Entry", "MA Ratio @ Entry", "BB %B @ Entry", "Vol % @ Entry"
    ])
    
    return {
        "equity_curve": equity_df,
        "trade_log": trade_df,
        "initial_capital": initial_capital
    }
