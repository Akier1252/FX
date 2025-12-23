import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from config import OANDA_ACCESS_TOKEN, OANDA_ENV, SYMBOL

# =========================================================
# AIモデル定義（ここも共通）
# =========================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

class MoE_Trader_AI(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_experts=5):
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
# =========================================================

# 設定
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

def evolve_ai():
    print("=== AI進化プロセス（夜間学習）を開始します ===")
    
    # 1. 直近データの取得（過去24時間分などを取得して微調整する）
    # OANDAから最新データを取得
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENV)
    oanda_symbol = SYMBOL.replace("XAUUSD", "XAU_USD")
    
    # 過去1000本（直近の相場傾向）を取得
    params = {"granularity": "M1", "count": 1000, "price": "M"}
    
    try:
        r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
        client.request(r)
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return

    # データ整形
    data = []
    for candle in r.response['candles']:
        if candle['complete']:
            data.append({
                "open": float(candle['mid']['o']),
                "high": float(candle['mid']['h']),
                "low": float(candle['mid']['l']),
                "close": float(candle['mid']['c']),
                "volume": int(candle['volume'])
            })
    df = pd.DataFrame(data)

    # 特徴量作成（ここも同じロジック）
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    df['volatility'] = df['high'] - df['low']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 正解ラベル（この直近データでどう動いたか？）
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)

    # 2. モデルとスケーラーの読み込み
    try:
        model = MoE_Trader_AI(input_dim=10)
        model.load_state_dict(torch.load(MODEL_PATH))
        # 転移学習モード（継続学習）
        model.train() 
        
        # 既存のスケーラーを読み込む（基準をぶらさないため）
        scaler = joblib.load(SCALER_PATH)
    except:
        print("既存モデルが見つかりません。まずは02_AI学習.pyを実行してください。")
        return

    # 3. 再学習（ファインチューニング）
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'diff_ma', 'volatility', 'rsi']
    X = df[feature_cols].values
    y = df['target'].values
    
    # 新しいデータも正規化
    X = scaler.transform(X) # fitではなくtransformを使うのがコツ
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # オプティマイザ設定（学習率を少し下げて、既存の知識を壊さないようにする）
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 通常より小さく
    criterion = nn.CrossEntropyLoss()

    print(">>> 最新相場への適応を開始（ファインチューニング）...")
    epochs = 20 # 回数は少なくて良い
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"適応進捗 {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 4. 上書き保存
    torch.save(model.state_dict(), MODEL_PATH)
    print("\n★ AI進化完了！モデルが最新の相場に適応しました。")
    print("明日もこの強化された 'best_ai_model.pth' でトレードします。")

if __name__ == "__main__":
    evolve_ai()