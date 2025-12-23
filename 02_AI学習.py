import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 3クラス対応 H-MoE モデル
# ==========================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

class MoE_Trader_AI(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_experts=5):
        # output_dim=3 (Stay, Buy, Sell)
        super(MoE_Trader_AI, self).__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x):
        weights = self.gate(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        weights = weights.unsqueeze(2)
        final_output = torch.sum(expert_outputs * weights, dim=1)
        return final_output

def train_ai():
    try:
        df = pd.read_csv("training_data.csv")
    except:
        print("CSVがありません。01_データ収集.pyを実行してください。")
        return

    # 使用する特徴量 (ATR, LogRetを追加)
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'diff_ma', 'atr', 'log_ret', 'rsi']
    
    X = df[feature_cols].values
    y = df['target'].values

    # 正規化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Tensor変換
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # モデル構築 (input=11, output=3)
    model = MoE_Trader_AI(input_dim=len(feature_cols), output_dim=3)
    
    # 損失関数 (クラス不均衡がある場合はweightを設定すると良いが、一旦標準で)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- 3クラス分類学習開始 (Stay/Buy/Sell) ---")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            # 評価
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_t)
                _, predicted = torch.max(test_out, 1)
                acc = (predicted == y_test_t).sum().item() / len(y_test_t)
            print(f"Epoch {epoch+1}: Loss {loss.item():.4f} | Test Acc: {acc:.2%}")

    torch.save(model.state_dict(), "best_ai_model.pth")
    print("★モデル保存完了: best_ai_model.pth")

if __name__ == "__main__":
    train_ai()