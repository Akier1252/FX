import time
import MetaTrader5 as mt5
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import copy

# 設定ファイル読み込み
from config import (
    OANDA_ACCESS_TOKEN, OANDA_ENV, SYMBOL, 
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL
)

# =========================================================
# 【設定】無限サイクルの時間割
# =========================================================
TRADE_DURATION_MIN = 60  # 何分間トレードしたら学習するか（例: 60分）
LEARNING_EPOCHS = 10     # 学習の回数（短時間で終わらせるため少なめに）
LOTS = 0.01              # 取引ロット数

# ファイルパス
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

# =========================================================
# AI脳みそ設計図（H-MoE）
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
# 機能関数群
# =========================================================

def initialize_mt5():
    if not mt5.initialize():
        print("MT5初期化失敗")
        return False
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5ログイン失敗: {mt5.last_error()}")
        return False
    return True

def feature_engineering(df):
    """データフレームから特徴量を作成する共通関数"""
    df = df.copy()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma20'] = df['ma20'].replace(0, np.nan)
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    df['volatility'] = df['high'] - df['low']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_mt5_features():
    """MT5から最新データを取得"""
    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M1, 0, 50)
    if rates is None or len(rates) == 0: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df = feature_engineering(df)
    
    latest = df.iloc[-1]
    features = [
        latest['open'], latest['high'], latest['low'], latest['close'], latest['tick_volume'],
        latest['ma5'], latest['ma20'], latest['diff_ma'], latest['volatility'], latest['rsi']
    ]
    return np.array(features).reshape(1, -1)

def execute_trade(action, confidence):
    """MT5で注文を出す"""
    if action == 0: return # 今回は買い(1)のみ
    
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None: return
    price = tick.ask
    point = mt5.symbol_info(MT5_SYMBOL).point
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - 500 * point,
        "tp": price + 1000 * point,
        "magic": 777777,
        "comment": f"Auto_AI_Conf:{confidence:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)
    print(f"  >>> 注文実行 (Confidence: {confidence:.2%})")

# =========================================================
# サイクル処理
# =========================================================

def phase_trade(model, scaler, end_time):
    """【戦闘モード】指定時間までトレードを行う"""
    print(f"\n[戦闘モード] {end_time.strftime('%H:%M')} まで市場を監視します...")
    
    while datetime.now() < end_time:
        try:
            features = get_mt5_features()
            if features is not None:
                feat_scaled = scaler.transform(features)
                feat_tensor = torch.FloatTensor(feat_scaled)
                
                with torch.no_grad():
                    output = model(feat_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    action = torch.argmax(probs, dim=1).item()
                    conf = probs[0][action].item()
                
                print(f"\r  AI視点: {'BUY' if action==1 else 'STAY'} (確信度:{conf:.1%})", end="")
                
                # 確信度85%以上でエントリー
                if action == 1 and conf > 0.85:
                    print("\n  ★ チャンス検知！")
                    execute_trade(action, conf)
                    time.sleep(60) # 連打防止
            
            time.sleep(1) # 1秒ごとに監視
            
        except Exception as e:
            print(f"  監視エラー(無視して継続): {e}")
            time.sleep(5)

def phase_evolution(model):
    """【進化モード】直近データを学習し、モデルを更新する"""
    print("\n\n[進化モード] 最新データを吸収し、脳を強化中...")
    
    # 1. OANDAから直近データの取得（過去1000分＝約16時間分）
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENV)
    params = {"granularity": "M1", "count": 1000, "price": "M"}
    oanda_symbol = SYMBOL.replace("XAUUSD", "XAU_USD")
    
    try:
        r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
        client.request(r)
    except Exception as e:
        print(f"  データ取得失敗: {e}")
        return model, False # 更新失敗

    data = []
    for c in r.response['candles']:
        if c['complete']:
            data.append({
                "open": float(c['mid']['o']), "high": float(c['mid']['h']),
                "low": float(c['mid']['l']), "close": float(c['mid']['c']),
                "volume": int(c['volume'])
            })
    df = pd.DataFrame(data)
    df = feature_engineering(df)
    
    # 正解ラベル作成 (未来が上がったか？)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)
    
    # 2. 学習準備
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'diff_ma', 'volatility', 'rsi']
    X = df[feature_cols].values
    y = df['target'].values
    
    # スケーラー読み込み（基準統一）
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # 3. ファインチューニング（微調整学習）
    model.train() # 学習モードへ
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 低学習率で慎重に更新
    criterion = nn.CrossEntropyLoss()
    
    initial_loss = 0
    final_loss = 0
    
    for epoch in range(LEARNING_EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch == 0: initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"  学習完了: 誤差 {initial_loss:.4f} -> {final_loss:.4f}")
    
    # 4. 保存
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval() # 推論モードに戻す
    
    return model, True

# =========================================================
# メイン：無限ループ
# =========================================================
def main():
    if not initialize_mt5(): return
    
    print("=== 完全自律型 AIトレーダーシステム起動 ===")
    print("これより「戦闘」と「進化」を永久に繰り返します。")

    # 初回モデルロード
    try:
        model = MoE_Trader_AI(input_dim=10)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        print("初期モデルロード完了。")
    except:
        print("エラー: モデルファイルがありません。まずは'02_AI学習.py'を実行してください。")
        return

    # 無限ループ開始
    cycle_count = 1
    try:
        while True:
            print(f"\n--- サイクル {cycle_count} 開始 ---")
            
            # 1. 戦闘フェーズ
            trade_end_time = datetime.now() + timedelta(minutes=TRADE_DURATION_MIN)
            phase_trade(model, scaler, trade_end_time)
            
            # 2. 進化フェーズ
            model, success = phase_evolution(model)
            if success:
                print(f"  >>> AIレベルアップ完了 (サイクル {cycle_count})")
            
            cycle_count += 1
            
    except KeyboardInterrupt:
        print("\n停止しました。")
        mt5.shutdown()

if __name__ == "__main__":
    main()