# models.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# RNN models
# -----------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# -----------------------
# Utilities to save/load PyTorch models and scalers
# -----------------------
def save_torch_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_torch_model(model_cls, path, device='cpu', **kwargs):
    model = model_cls(**kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
