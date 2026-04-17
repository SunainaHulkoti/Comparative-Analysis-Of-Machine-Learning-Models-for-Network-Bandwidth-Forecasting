# train_models.py
from utils import fetch_infra, ensure_dir
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import GRUModel, LSTMModel, save_torch_model, save_scaler
from utils import fetch_infra, ensure_dir
import joblib
import os

ARTIFACT_DIR = "artifacts"
ensure_dir(ARTIFACT_DIR)

# 1) Fetch historical data from infra collection
df = fetch_infra(limit=5000)  # fetch recent N
if df.empty:
    raise SystemExit("No infra data available for training. Insert logs first.")

# We'll use bandwidth_in for training (you can extend to out)
df = df.dropna(subset=['bandwidth_in'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['ds'] = df['timestamp']
df['y'] = df['bandwidth_in'].astype(float)

# ---------------- Prophet ----------------
print("Training Prophet...")
prophet_df = df[['ds','y']].rename(columns={'ds':'ds','y':'y'})
m = Prophet()
m.fit(prophet_df)
# save Prophet model using joblib (prophet is picklable)
joblib.dump(m, os.path.join(ARTIFACT_DIR, "prophet_model.pkl"))
print("Saved Prophet model.")

# ---------------- Prepare for RNNs ----------------
SEQ_LEN = 5
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df['y'].values.reshape(-1,1)).flatten()
save_scaler(scaler, os.path.join(ARTIFACT_DIR, "scaler_in.pkl"))

def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr)-seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)
if len(X) < 10:
    print("Not enough data for GRU/LSTM training; skipping RNN training.")
else:
    # convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [N, seq, 1]
    y_t = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # train GRU
    print("Training GRU...")
    gru = GRUModel(input_dim=1, hidden_dim=64)
    opt = torch.optim.Adam(gru.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(30):
        total_loss = 0.0
        gru.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = gru(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"GRU epoch {epoch+1}, loss {total_loss/len(loader):.6f}")
    save_torch_model(gru, os.path.join(ARTIFACT_DIR, "gru_in.pt"))
    print("Saved GRU model.")

    # train LSTM
    print("Training LSTM...")
    lstm = LSTMModel(input_dim=1, hidden_dim=64)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.01)
    for epoch in range(30):
        total_loss = 0.0
        lstm.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = lstm(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"LSTM epoch {epoch+1}, loss {total_loss/len(loader):.6f}")
    save_torch_model(lstm, os.path.join(ARTIFACT_DIR, "lstm_in.pt"))
    print("Saved LSTM model.")

print("Training finished. Artifacts saved to:", ARTIFACT_DIR)
