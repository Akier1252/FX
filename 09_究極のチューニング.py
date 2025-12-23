import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os

# チューニング用モデル定義 (パラメータ可変)
class TunableExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(TunableExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class TunableGating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate):
        super(TunableGating, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

class TunableMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, dropout_rate):
        super(TunableMoE, self).__init__()
        self.experts = nn.ModuleList([
            TunableExpert(input_dim, hidden_dim, output_dim, dropout_rate) for _ in range(num_experts)
        ])
        self.gate = TunableGating(input_dim, num_experts, dropout_rate)
    def forward(self, x):
        weights = self.gate(x)
        exp_out = torch.stack([e(x) for e in self.experts], dim=1)
        return torch.sum(exp_out * weights.unsqueeze(2), dim=1)

def objective(trial):
    df = pd.read_csv("training_data.csv")
    cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'diff_ma', 'atr', 'log_ret', 'rsi']
    X_all = df[cols].values
    y_all = df['target'].values

    # パラメータ探索範囲
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    num_experts = trial.suggest_int("num_experts", 3, 10)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, test_idx in tscv.split(X_all):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Output Dim = 3 (Stay, Buy, Sell)
        model = TunableMoE(len(cols), hidden_dim, 3, num_experts, dropout)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        
        for _ in range(15): # 高速化のためエポック少なめ
            opt.zero_grad()
            out = model(torch.FloatTensor(X_tr))
            loss = crit(out, torch.LongTensor(y_tr))
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            model.eval()
            out = model(torch.FloatTensor(X_te))
            pred = torch.argmax(out, 1)
            acc = (pred == torch.LongTensor(y_te)).sum().item() / len(y_te)
            scores.append(acc)
            
    return np.mean(scores)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best Params:", study.best_params)