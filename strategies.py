import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from indicators import add_indicators
from ml_model import train_and_predict_ml_signals
from lstm_model import train_and_predict_lstm_signals
from volatility_model import fit_garch_volatility

STRATEGY_OPTIONS = [
    "MA Crossover",
    "RSI Threshold",
    "Bollinger Reversion",
    "AI Hybrid (Random Forest + GARCH)",
    "AI Hybrid (XGBoost + GARCH)",
    "LSTM Sequence Model (Comparison Benchmark)"
]

def generate_ma_crossover_signals(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.Series:
    """Generate signals for Moving Average Crossover strategy."""
    data = add_indicators(df, fast_ma=fast_period, slow_ma=slow_period)
    raw_signal = np.where(data['MA_Fast'] > data['MA_Slow'], 1, -1)
    signals = pd.Series(raw_signal, index=data.index)
    signals.iloc[:slow_period] = 0
    return signals

def generate_rsi_signals(df: pd.DataFrame, rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """Generate signals for RSI Mean Reversion strategy."""
    data = add_indicators(df, rsi_period=rsi_period)
    signals = pd.Series(0, index=data.index)
    signals[data['RSI'] < oversold] = 1
    signals[data['RSI'] > overbought] = -1
    return signals

def generate_bollinger_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0) -> pd.Series:
    """Generate signals for Bollinger Band Reversion strategy."""
    data = add_indicators(df, bb_period=bb_period, bb_std=bb_std)
    signals = pd.Series(0, index=data.index)
    signals[data['Close'] < data['BB_Lower']] = 1
    signals[data['Close'] > data['BB_Upper']] = -1
    return signals

def generate_signals(
    df: pd.DataFrame,
    strategy_name: str,
    params: Dict[str, Any]
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Unified signal generation wrapper for all strategy types.
    """
    metadata: Dict[str, Any] = {"strategy": strategy_name, "params": params}
    df_featured = add_indicators(df)
    
    # Fit GARCH(1,1) volatility layer for all AI strategies
    garch_df, garch_meta = fit_garch_volatility(
        df_featured,
        multiplier=params.get("garch_multiplier", 2.0)
    )
    metadata["garch"] = garch_meta
    
    if strategy_name == "MA Crossover":
        fast = params.get("fast_period", 20)
        slow = params.get("slow_period", 50)
        signals = generate_ma_crossover_signals(df, fast_period=fast, slow_period=slow)
        
    elif strategy_name == "RSI Threshold":
        rsi_period = params.get("rsi_period", 14)
        oversold = params.get("oversold", 30.0)
        overbought = params.get("overbought", 70.0)
        signals = generate_rsi_signals(df, rsi_period=rsi_period, oversold=oversold, overbought=overbought)
        
    elif strategy_name == "Bollinger Reversion":
        bb_period = params.get("bb_period", 20)
        bb_std = params.get("bb_std", 2.0)
        signals = generate_bollinger_signals(df, bb_period=bb_period, bb_std=bb_std)
        
    elif "Random Forest" in strategy_name or strategy_name == "AI Model (ML Classifier)":
        horizon = params.get("horizon", 5)
        confidence = params.get("confidence", 0.55)
        signals, ml_meta = train_and_predict_ml_signals(
            df_featured,
            horizon=horizon,
            confidence_threshold=confidence,
            model_type="Random Forest"
        )
        metadata.update(ml_meta)
        
    elif "XGBoost" in strategy_name:
        horizon = params.get("horizon", 5)
        confidence = params.get("confidence", 0.55)
        signals, ml_meta = train_and_predict_ml_signals(
            df_featured,
            horizon=horizon,
            confidence_threshold=confidence,
            model_type="XGBoost"
        )
        metadata.update(ml_meta)
        
    elif "LSTM" in strategy_name:
        horizon = params.get("horizon", 5)
        confidence = params.get("confidence", 0.55)
        seq_length = params.get("seq_length", 10)
        signals, lstm_meta = train_and_predict_lstm_signals(
            df_featured,
            seq_length=seq_length,
            horizon=horizon,
            confidence_threshold=confidence
        )
        metadata.update(lstm_meta)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
        
    return signals, metadata
