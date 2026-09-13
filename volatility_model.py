import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def fit_garch_volatility(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    multiplier: float = 2.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a GARCH(p, q) model on stock return series and calculate conditional volatility,
    GARCH price bands, and dynamic risk parameters.
    
    Parameters:
        df (pd.DataFrame): Stock OHLCV DataFrame with 'Close' price.
        p (int): GARCH lag order (default 1).
        q (int): ARCH lag order (default 1).
        multiplier (float): Multiplier for GARCH volatility bands (default 2.0).
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame with added columns: 'GARCH_Vol_Daily', 'GARCH_Vol_Ann', 
              'GARCH_Upper', 'GARCH_Lower', 'Dynamic_Stop_Loss_Pct'
            - Model summary metadata dictionary
    """
    data = df.copy()
    
    # Calculate daily percentage returns (scaled by 100 for arch model convergence)
    returns = 100.0 * data['Close'].pct_change().dropna()
    
    garch_vol_daily = pd.Series(index=data.index, dtype=float)
    fit_status = "Fallback EWMA (arch library unavailable)"
    
    try:
        from arch import arch_model
        
        # Fit GARCH(1,1) model with Constant mean & Normal distribution
        am = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant', dist='normal')
        res = am.fit(disp='off', show_warning=False)
        
        # Conditional volatility in percentage scale -> convert back to decimal
        cond_vol = res.conditional_volatility / 100.0
        garch_vol_daily.update(cond_vol)
        fit_status = f"GARCH({p},{q}) fitted successfully (AIC: {res.aic:.2f})"
        
    except Exception as e:
        # Fallback to 20-day Exponentially Weighted Moving Average (EWMA) volatility
        ewma_vol = data['Close'].pct_change().ewm(span=20).std()
        garch_vol_daily.update(ewma_vol)
        fit_status = f"EWMA Volatility Fallback (Reason: {str(e)})"
        
    # Fill early missing rows with rolling std
    garch_vol_daily = garch_vol_daily.bfill().ffill()
    garch_vol_daily = garch_vol_daily.fillna(data['Close'].pct_change().std() or 0.015)
    
    # Annualized Volatility (assuming 252 trading days)
    garch_vol_ann = garch_vol_daily * np.sqrt(252)
    
    # GARCH Volatility Price Bands
    garch_upper = data['Close'] * (1.0 + multiplier * garch_vol_daily)
    garch_lower = data['Close'] * (1.0 - multiplier * garch_vol_daily)
    
    # Dynamic Risk Parameters
    # Dynamic stop loss = k * daily volatility (e.g. 2 * daily_vol)
    dynamic_stop_loss = (multiplier * garch_vol_daily).clip(lower=0.01, upper=0.15)
    
    data['GARCH_Vol_Daily'] = garch_vol_daily
    data['GARCH_Vol_Ann'] = garch_vol_ann
    data['GARCH_Upper'] = garch_upper
    data['GARCH_Lower'] = garch_lower
    data['Dynamic_Stop_Loss_Pct'] = dynamic_stop_loss
    
    metadata = {
        "status": fit_status,
        "latest_daily_vol_pct": float(garch_vol_daily.iloc[-1] * 100.0),
        "latest_ann_vol_pct": float(garch_vol_ann.iloc[-1] * 100.0),
        "p": p,
        "q": q,
        "band_multiplier": multiplier
    }
    
    return data, metadata
