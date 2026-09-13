import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any

POPULAR_NSE_STOCKS: Dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "INFY.NS": "Infosys Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "TATASTEEL.NS": "Tata Steel Ltd",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Ltd",
    "ITC.NS": "ITC Ltd",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd",
    "WIPRO.NS": "Wipro Ltd",
    "LT.NS": "Larsen & Toubro Ltd",
    "AXISBANK.NS": "Axis Bank Ltd",
    "MARUTI.NS": "Maruti Suzuki India Ltd",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd"
}

def format_ticker_symbol(ticker: str) -> str:
    """Ensure the ticker symbol has proper NSE format (.NS)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return "RELIANCE.NS"
    if not ticker.endswith(".NS") and not ticker.startswith("^"):
        ticker = f"{ticker}.NS"
    return ticker

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker: str, period: str = "1y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch historical daily OHLCV stock data via yfinance with forward-fill error handling.
    """
    formatted_ticker = format_ticker_symbol(ticker)
    
    try:
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=period, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No price data found for ticker '{formatted_ticker}'. Please check the symbol.")
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in data.")
            
        df = df[cols].copy()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Section 11 Error Handling: Check & forward-fill missing values from trading halts
        had_nans = df.isnull().values.any()
        if had_nans:
            df = df.ffill().bfill()
            
        df.dropna(subset=['Close'], inplace=True)
        
        info = {}
        try:
            raw_info = stock.info or {}
            info['shortName'] = raw_info.get('shortName', POPULAR_NSE_STOCKS.get(formatted_ticker, formatted_ticker))
            info['symbol'] = formatted_ticker
            info['currency'] = raw_info.get('currency', 'INR')
            info['fiftyTwoWeekHigh'] = raw_info.get('fiftyTwoWeekHigh', df['Close'].max())
            info['fiftyTwoWeekLow'] = raw_info.get('fiftyTwoWeekLow', df['Close'].min())
        except Exception:
            info = {
                'shortName': POPULAR_NSE_STOCKS.get(formatted_ticker, formatted_ticker),
                'symbol': formatted_ticker,
                'currency': 'INR',
                'fiftyTwoWeekHigh': df['Close'].max(),
                'fiftyTwoWeekLow': df['Close'].min()
            }
            
        info['data_gap_filled'] = had_nans
        info['num_bars'] = len(df)
        
        return df, info
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock data for {formatted_ticker}: {str(e)}")
