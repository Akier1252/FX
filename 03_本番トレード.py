import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
from datetime import datetime
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL

# =========================================================
# AIの脳みそ設計図（読み込みエラー防止のためここにも記述）
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

# ファイルパス設定
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"
LOTS = 0.01  # 1回の取引量（1万円運用の場合は0.01推奨）

def initialize_mt5():
    # MT5に接続
    if not mt5.initialize():
        print("MT5の起動に失敗しました。")
        return False
    
    # ログイン
    authorized = mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print(f"MT5へのログインに失敗しました: {mt5.last_error()}")
        return False
    
    print(f"MT5ログイン成功: 口座 {MT5_LOGIN}")
    return True

def get_latest_features():
    """MT5から直近データを取得し、AIに入力できる形にする"""
    # 過去50本取得
    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M1, 0, 50)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 指標計算（データ収集時と全く同じ計算式にする必要がある）
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    # 0除算回避
    df['ma20'] = df['ma20'].replace(0, np.nan)
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    
    df['volatility'] = df['high'] - df['low']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 欠損処理
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True) 

    # 最新の1行を取得
    latest = df.iloc[-1]
    
    # AIに入力する順番
    features = [
        latest['open'], latest['high'], latest['low'], latest['close'], latest['tick_volume'],
        latest['ma5'], latest['ma20'], latest['diff_ma'], latest['volatility'], latest['rsi']
    ]
    return np.array(features).reshape(1, -1)

def execute_trade(action):
    # 今回は買い(1)のみ実装
    if action == 0:
        return 

    # 現在価格を取得
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None:
        return
    
    price = tick.ask
    point = mt5.symbol_info(MT5_SYMBOL).point
    
    # 注文リクエスト（損切り・利確付き）
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - 500 * point, # 損切り (例: ゴールドで50pips)
        "tp": price + 1000 * point, # 利確 (例: ゴールドで100pips)
        "magic": 999000,
        "comment": "AI_MoE_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f">>> 買い注文が約定しました！ 価格: {price}")
    else:
        print(f"注文エラー: {result.comment}")

def main():
    if not initialize_mt5():
        return

    # モデルとスケーラーの読み込み
    try:
        model = MoE_Trader_AI(input_dim=10)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # 推論モード
        
        scaler = joblib.load(SCALER_PATH)
        print("\n=== AIトレーダーシステム起動完了 ===")
        print("市場の監視を開始します...")
    except Exception as e:
        print(f"\n【エラー】必要なファイルが見つかりません: {e}")
        print("'02_AI学習.py' を実行してモデルを作成してください。")
        return

    try:
        while True:
            # 最新データの取得
            features = get_latest_features()
            
            if features is not None:
                # 正規化
                feat_scaled = scaler.transform(features)
                feat_tensor = torch.FloatTensor(feat_scaled)
                
                # AIによる予測
                with torch.no_grad():
                    output = model(feat_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    
                    # 確率が高い方の行動を選ぶ
                    action = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][action].item()
                
                # 画面表示
                prediction_str = "BUY (買い)" if action == 1 else "STAY (様子見)"
                print(f"\r現在判定: {prediction_str} | AI確信度: {confidence:.1%}", end="")
                
                # 条件: 買い判定 かつ 確信度が80%以上
                if action == 1 and confidence > 0.8:
                    print("\n★ 高利益チャンスを検知しました！")
                    execute_trade(action)
                    time.sleep(60) # 連続注文を防ぐため1分待つ
            
            # 1秒待機
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nシステムを停止しました。")
        mt5.shutdown()

if __name__ == "__main__":
    main()