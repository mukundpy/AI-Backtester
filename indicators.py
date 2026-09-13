import pandas as pd
import numpy as np

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def add_indicators(
    df: pd.DataFrame,
    fast_ma: int = 20,
    slow_ma: int = 50,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate and append technical indicators to the DataFrame.
    
    Returns a copy of DataFrame with added columns:
    - MA_Fast, MA_Slow
    - RSI
    - BB_Upper, BB_Middle, BB_Lower, BB_PctB
    - Daily_Return
    - Rolling_Vol_20
    - Volume_SMA_20
    - Volume_Ratio
    """
    data = df.copy()
    
    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Moving Averages
    data['MA_Fast'] = data['Close'].rolling(window=fast_ma).mean()
    data['MA_Slow'] = data['Close'].rolling(window=slow_ma).mean()
    data['MA_Ratio'] = data['MA_Fast'] / data['MA_Slow']
    
    # RSI
    data['RSI'] = calculate_rsi(data['Close'], period=rsi_period)
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
    bb_rolling_std = data['Close'].rolling(window=bb_period).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * bb_rolling_std)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * bb_rolling_std)
    
    # %B = (Price - Lower) / (Upper - Lower)
    bb_range = (data['BB_Upper'] - data['BB_Lower']).replace(0, np.nan)
    data['BB_PctB'] = (data['Close'] - data['BB_Lower']) / bb_range
    
    # Rolling Volatility (20-day annualized)
    data['Rolling_Vol_20'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Volume indicators
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20'].replace(0, np.nan)
    
    return data
