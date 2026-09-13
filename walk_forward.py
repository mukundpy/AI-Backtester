import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ml_model import FEATURE_COLS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_walk_forward_validation(
    df: pd.DataFrame,
    train_window: int = 250,  # ~1 trading year
    test_window: int = 60,    # ~3 trading months
    horizon: int = 5,
    confidence_threshold: float = 0.55
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Execute walk-forward rolling validation (train on rolling window -> test on next window -> slide forward).
    
    Returns:
        Tuple[pd.Series, Dict[str, Any]]:
            - Out-of-sample signal series constructed sequentially across walk-forward windows.
            - Validation metadata including rolling window accuracies and feature importance drift.
    """
    data = df.copy()
    data['Future_Return'] = data['Close'].shift(-horizon) / data['Close'] - 1.0
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    valid_data = data.dropna(subset=FEATURE_COLS + ['Target']).copy()
    
    total_bars = len(valid_data)
    if total_bars < (train_window + test_window):
        # Fallback if historical data is shorter than combined window
        train_window = max(50, int(total_bars * 0.6))
        test_window = max(20, int(total_bars * 0.2))
        
    out_of_sample_signals = pd.Series(0, index=df.index)
    window_results = []
    feature_importance_history = {col: [] for col in FEATURE_COLS}
    
    start_idx = 0
    while (start_idx + train_window + test_window) <= total_bars:
        train_df = valid_data.iloc[start_idx : start_idx + train_window]
        test_df = valid_data.iloc[start_idx + train_window : start_idx + train_window + test_window]
        
        X_train = train_df[FEATURE_COLS]
        y_train = train_df['Target']
        X_test = test_df[FEATURE_COLS]
        y_test = test_df['Target']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        test_acc = float(model.score(X_test_scaled, y_test)) if len(y_test) > 0 else 0.0
        train_acc = float(model.score(X_train_scaled, y_train))
        
        # Track feature importance drift
        for col, imp in zip(FEATURE_COLS, model.feature_importances_):
            feature_importance_history[col].append(imp)
            
        probs = model.predict_proba(X_test_scaled)[:, 1]
        test_signals = np.where(probs >= confidence_threshold, 1,
                       np.where(probs <= (1.0 - confidence_threshold), -1, 0))
                       
        out_of_sample_signals.update(pd.Series(test_signals, index=test_df.index))
        
        window_results.append({
            "window_start": train_df.index[0].strftime('%Y-%m-%d'),
            "train_end": train_df.index[-1].strftime('%Y-%m-%d'),
            "test_end": test_df.index[-1].strftime('%Y-%m-%d'),
            "train_acc": train_acc,
            "test_acc": test_acc
        })
        
        start_idx += test_window  # Slide window forward
        
    mean_test_acc = float(np.mean([w['test_acc'] for w in window_results])) if window_results else 0.0
    mean_train_acc = float(np.mean([w['train_acc'] for w in window_results])) if window_results else 0.0
    
    importance_stability = {
        col: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for col, vals in feature_importance_history.items() if len(vals) > 0
    }
    
    metadata = {
        "num_windows": len(window_results),
        "mean_train_acc": mean_train_acc,
        "mean_test_acc": mean_test_acc,
        "window_results": window_results,
        "importance_stability": importance_stability
    }
    
    return out_of_sample_signals, metadata
