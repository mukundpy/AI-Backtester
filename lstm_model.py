import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

FEATURE_COLS = ['RSI', 'MA_Ratio', 'BB_PctB', 'Daily_Return', 'Rolling_Vol_20', 'Volume_Ratio']

def create_sequence_dataset(X: np.ndarray, y: np.ndarray, seq_length: int = 10):
    """Convert 2D array of features into 3D sequence arrays for LSTM processing."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def train_and_predict_lstm_signals(
    df: pd.DataFrame,
    seq_length: int = 10,
    horizon: int = 5,
    confidence_threshold: float = 0.55,
    train_ratio: float = 0.7,
    epochs: int = 25
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Train an LSTM sequence model on historical feature sequences (chronological split)
    and output out-of-sample signals and benchmark comparison metrics.
    """
    data = df.copy()
    data['Future_Return'] = data['Close'].shift(-horizon) / data['Close'] - 1.0
    data['Target'] = (data['Future_Return'] > 0).astype(int)
    
    valid_data = data.dropna(subset=FEATURE_COLS + ['Target']).copy()
    
    if len(valid_data) < (seq_length + 40):
        signals = pd.Series(0, index=df.index)
        return signals, {"error": "Insufficient data to train LSTM model (minimum 60 data points required)."}
        
    X_raw = valid_data[FEATURE_COLS].values
    y_raw = valid_data['Target'].values
    dates_raw = valid_data.index
    
    # Chronological Split
    split_idx = int(len(valid_data) * train_ratio)
    
    # Normalize features using scaler fitted on training data only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_raw[:split_idx])
    X_scaled = scaler.transform(X_raw)
    
    # Create sequence dataset
    X_seq, y_seq = create_sequence_dataset(X_scaled, y_raw, seq_length=seq_length)
    target_dates = dates_raw[seq_length:]
    
    seq_split_idx = max(1, split_idx - seq_length)
    X_train_seq = X_seq[:seq_split_idx]
    y_train_seq = y_seq[:seq_split_idx]
    X_test_seq = X_seq[seq_split_idx:]
    y_test_seq = y_seq[seq_split_idx:]
    
    probs = np.full(len(y_seq), 0.5)
    model_type = "PyTorch LSTM"
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        torch.manual_seed(42)
        
        class PyTorchLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=32, num_layers=1):
                super(PyTorchLSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return self.sigmoid(out)
                
        input_dim = X_seq.shape[2]
        model = PyTorchLSTM(input_dim=input_dim, hidden_dim=32)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            X_full_t = torch.tensor(X_seq, dtype=torch.float32)
            probs = model(X_full_t).numpy().flatten()
            
    except Exception as e:
        # Fallback to 1D Neural Net / MLP if PyTorch training is interrupted or unavailable
        model_type = f"MLP Sequence Net (Fallback: {str(e)[:30]})"
        from sklearn.neural_network import MLPClassifier
        X_flat_train = X_train_seq.reshape(len(X_train_seq), -1)
        X_flat_full = X_seq.reshape(len(X_seq), -1)
        
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
        mlp.fit(X_flat_train, y_train_seq)
        probs = mlp.predict_proba(X_flat_full)[:, 1]
        
    # Evaluate Out-of-Sample Accuracy
    if len(y_test_seq) > 0:
        test_probs = probs[seq_split_idx:]
        test_preds = (test_probs >= 0.5).astype(int)
        test_acc = float(np.mean(test_preds == y_test_seq))
    else:
        test_acc = 0.0
        
    train_probs = probs[:seq_split_idx]
    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = float(np.mean(train_preds == y_train_seq)) if len(y_train_seq) > 0 else 0.0
    
    # Generate discrete signals
    raw_signals = np.where(probs >= confidence_threshold, 1,
                  np.where(probs <= (1.0 - confidence_threshold), -1, 0))
                  
    signal_series = pd.Series(raw_signals, index=target_dates)
    full_signals = pd.Series(0, index=df.index)
    full_signals.update(signal_series)
    
    split_date = dates_raw[split_idx].strftime('%Y-%m-%d') if split_idx < len(dates_raw) else 'N/A'
    
    metadata = {
        "model_type": model_type,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "split_date": split_date,
        "train_samples": len(X_train_seq),
        "test_samples": len(X_test_seq),
        "seq_length": seq_length,
        "confidence_threshold": confidence_threshold
    }
    
    return full_signals, metadata
