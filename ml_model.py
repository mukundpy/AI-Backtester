import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

FEATURE_COLS = ['RSI', 'MA_Ratio', 'BB_PctB', 'Daily_Return', 'Rolling_Vol_20', 'Volume_Ratio']

def train_and_predict_ml_signals(
    df: pd.DataFrame,
    horizon: int = 5,
    confidence_threshold: float = 0.55,
    model_type: str = "Random Forest",
    train_ratio: float = 0.7
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Train an ML classifier (Random Forest, XGBoost, Logistic Regression) on technical features
    using a strict time-respecting train/test split to prevent look-ahead bias.
    """
    data = df.copy()
    
    # 1. Target Label Creation: 1 if price after N days is higher than today, else 0
    data['Future_Return'] = data['Close'].shift(-horizon) / data['Close'] - 1.0
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    # Drop rows missing features or target label
    valid_data = data.dropna(subset=FEATURE_COLS + ['Target']).copy()
    
    if len(valid_data) < 50:
        signals = pd.Series(0, index=df.index)
        return signals, {"error": "Insufficient data to train ML model (minimum 50 data points required)."}
    
    # 2. Chronological Train/Test Split
    split_idx = int(len(valid_data) * train_ratio)
    train_df = valid_data.iloc[:split_idx]
    test_df = valid_data.iloc[split_idx:]
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['Target']
    X_full = valid_data[FEATURE_COLS]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)
    
    # 3. Model Fitting
    if model_type == "XGBoost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=42, eval_metric='logloss'
            )
            model.fit(X_train_scaled, y_train)
            importances = dict(zip(FEATURE_COLS, model.feature_importances_))
        except Exception:
            # Fallback to Random Forest if xgboost module import fails
            model_type = "Random Forest (XGB Fallback)"
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train_scaled, y_train)
            importances = dict(zip(FEATURE_COLS, model.feature_importances_))
            
    elif model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        if hasattr(model, 'coef_'):
            importances = dict(zip(FEATURE_COLS, model.coef_[0]))
        else:
            importances = {col: 0.0 for col in FEATURE_COLS}
    else:
        # Default: Random Forest
        model_type = "Random Forest"
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        importances = dict(zip(FEATURE_COLS, model.feature_importances_))
        
    # Evaluate Out-of-Sample Accuracy on test set
    if len(test_df) > 0:
        X_test_scaled = scaler.transform(test_df[FEATURE_COLS])
        test_acc = float(model.score(X_test_scaled, test_df['Target']))
    else:
        test_acc = 0.0
        
    train_acc = float(model.score(X_train_scaled, y_train))
    
    # 4. Generate Predictions & Probabilities
    probs = model.predict_proba(X_full_scaled)[:, 1]
    
    raw_signals = np.where(probs >= confidence_threshold, 1,
                  np.where(probs <= (1.0 - confidence_threshold), -1, 0))
                  
    signal_series = pd.Series(raw_signals, index=valid_data.index)
    
    full_signals = pd.Series(0, index=df.index)
    full_signals.update(signal_series)
    
    split_date = train_df.index[-1].strftime('%Y-%m-%d') if not train_df.empty else 'N/A'
    
    metadata = {
        "model_type": model_type,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "split_date": split_date,
        "feature_importances": importances,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "horizon": horizon,
        "confidence_threshold": confidence_threshold
    }
    
    return full_signals, metadata
