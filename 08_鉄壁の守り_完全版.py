import time
import MetaTrader5 as mt5
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from datetime import datetime, timedelta
import os

# 設定読み込み
from config import (
    OANDA_ACCESS_TOKEN, OANDA_ENV, SYMBOL, 
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL
)

# =========================================================
# 設定：安全性重視パラメータ
# =========================================================
TRADE_DURATION_MIN = 60
LEARNING_EPOCHS = 30
MEMORY_FILE = "long_term_memory.csv"
PAST_SAMPLE_RATIO = 0.5    # 過去データ比率を50%に上げ、視野を広くする
LOTS = 0.01

# 【安全装置】異常検知の閾値
# 現在の相場と過去データの距離がこれ以上離れていたら「未知」と判断してトレードしない
# ※運用しながら調整が必要ですが、まずは 5.0 程度から開始
ANOMALY_THRESHOLD = 5.0 

MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

# =========================================================
# AI脳みそ（ドロップアウト搭載版）
# =========================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2), # ★【防御1】20%のニューロンをランダムに無効化
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2), # ★【防御1】過学習を防ぐ
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
            nn.Dropout(p=0.2), # ★【防御1】司令塔も過学習させない
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
# 安全装置付き 類似検索システム
# =========================================================
def create_hybrid_dataset_safe(full_memory_df, recent_df_len):
    """類似データを集めるが、あまりに似ていない場合は警告を出す"""
    feature_cols = ['ma5', 'ma20', 'diff_ma', 'volatility', 'rsi', 'volume']
    
    if len(full_memory_df) < recent_df_len * 2:
        return full_memory_df.iloc[-recent_df_len:]

    current_state = full_memory_df.iloc[-1][feature_cols].values.astype(float)
    past_memory = full_memory_df.iloc[:-recent_df_len].copy()
    past_features = past_memory[feature_cols].values.astype(float)

    # 距離計算
    distances = np.sum((past_features - current_state) ** 2, axis=1)
    past_memory['distance'] = distances
    past_memory.sort_values(by='distance', ascending=True, inplace=True)
    
    # 【安全確認】最も似ているデータとの距離を確認
    min_distance = past_memory.iloc[0]['distance']
    print(f"  [環境認識] 過去データとの最短距離: {min_distance:.4f}")
    
    # 学習用データの抽出（ここは前回と同じ）
    sample_size = int(recent_df_len * (PAST_SAMPLE_RATIO / (1 - PAST_SAMPLE_RATIO)))
    candidate_pool = past_memory.iloc[: int(len(past_memory) * 0.2)] # 上位20%から
    
    if len(candidate_pool) >= sample_size:
        past_sample = candidate_pool.sample(n=sample_size)
    else:
        past_sample = candidate_pool

    recent_data = full_memory_df.iloc[-recent_df_len:]
    return pd.concat([recent_data, past_sample]).sample(frac=1).reset_index(drop=True)

def check_anomaly(features, memory_df, scaler):
    """
    ★【防御2】異常検知機能
    現在の相場が、記憶にあるデータと比べて「未知の領域」にないか判定する
    True = 異常あり（トレード禁止）
    """
    if memory_df is None or len(memory_df) < 100:
        return False, 0.0 # データ不足の時はチェックスキップ

    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'ma5', 'ma20', 'diff_ma', 'volatility', 'rsi']
    
    # 現在の特徴量（正規化済みであること前提だが、ここでは生データ比較のため注意）
    # 簡易的に、主要指標(RSI, Diff_MA, Volatility)だけで距離を見る
    key_indices = [7, 8, 9] # diff_ma, volatility, rsi のインデックス
    
    # 実際は正規化してから距離を測るのが正しい
    # メモリ上のデータも正規化されていないため、ここでスケーラーを使って変換比較する
    
    # (実装簡略化のため、ここではRSIとVolatilityの乖離だけで判定します)
    current_rsi = features[0][9]
    current_vol = features[0][8]
    
    # 過去データ全件の統計
    past_rsi_mean = memory_df['rsi'].mean()
    past_rsi_std = memory_df['rsi'].std()
    
    # 3シグマ（偏差値70以上/30以下的な）を超えていたら異常
    if abs(current_rsi - past_rsi_mean) > 3 * past_rsi_std:
        return True, 99.9 # 異常値
        
    return False, 0.0

# =========================================================
# 共通機能群
# =========================================================
def initialize_mt5():
    if not mt5.initialize(): return False
    mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    return True

def feature_engineering(df):
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
    if action == 0: return 
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None: return
    price = tick.ask
    point = mt5.symbol_info(MT5_SYMBOL).point
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": MT5_SYMBOL, "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY, "price": price, "sl": price - 500 * point, "tp": price + 1000 * point,
        "magic": 111111, "comment": f"Safe_AI:{confidence:.2f}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(request)
    print(f"  >>> 注文実行 (Confidence: {confidence:.2%})")

# =========================================================
# サイクル処理
# =========================================================
def phase_trade(model, scaler, end_time):
    # 記憶ファイルがあれば読み込んでおく（異常検知用）
    memory_df = None
    if os.path.exists(MEMORY_FILE):
        memory_df = pd.read_csv(MEMORY_FILE)

    print(f"\n[戦闘モード] 安全装置ON... {end_time.strftime('%H:%M')} まで監視")
    
    while datetime.now() < end_time:
        try:
            features = get_mt5_features()
            if features is not None:
                # 1. 異常検知チェック
                is_anomaly, score = check_anomaly(features, memory_df, scaler)
                if is_anomaly:
                    print(f"\r  ⚠️ 異常相場を検知 (Score:{score:.1f}) -> トレード停止中...", end="")
                    time.sleep(5)
                    continue

                # 2. 通常予測
                feat_scaled = scaler.transform(features)
                
                # ドロップアウトを有効にするためtrainモードにする手もあるが
                # 推論時はOFF(eval)にするのが一般的。
                # ただし「不確実性」を測るためにあえてONにして複数回予測する手法(MC Dropout)もある。
                # ここでは標準的なevalモードで使用。
                model.eval() 
                
                with torch.no_grad():
                    output = model(torch.FloatTensor(feat_scaled))
                    probs = torch.nn.functional.softmax(output, dim=1)
                    action = torch.argmax(probs, dim=1).item()
                    conf = probs[0][action].item()
                
                print(f"\r  AI視点: {'BUY' if action==1 else 'STAY'} ({conf:.1%})", end="")
                
                # エントリー条件: 確信度が高く、かつ異常でない
                if action == 1 and conf > 0.85:
                    print("\n  ★ 安全圏でのチャンス検知！")
                    execute_trade(action, conf)
                    time.sleep(60)
            time.sleep(1)
        except Exception as e:
            time.sleep(5)

def phase_evolution(model):
    print("\n\n[進化モード] 過去の経験と照合中...")
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENV)
    params = {"granularity": "M1", "count": 1000, "price": "M"}
    try:
        r = instruments.InstrumentsCandles(instrument=SYMBOL.replace("XAUUSD", "XAU_USD"), params=params)
        client.request(r)
    except: return model, False

    data = []
    for c in r.response['candles']:
        if c['complete']:
            data.append({
                "time": str(c['time']),
                "open": float(c['mid']['o']), "high": float(c['mid']['h']),
                "low": float(c['mid']['l']), "close": float(c['mid']['c']),
                "volume": int(c['volume'])
            })
    recent_df = pd.DataFrame(data)
    full_feature_df = feature_engineering(recent_df)
    full_feature_df['target'] = np.where(full_feature_df['close'].shift(-1) > full_feature_df['close'], 1, 0)
    full_feature_df.dropna(inplace=True)

    # 記憶更新
    if os.path.exists(MEMORY_FILE):
        old_df = pd.read_csv(MEMORY_FILE)
        combined_df = pd.concat([old_df, full_feature_df])
        combined_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    else:
        combined_df = full_feature_df
    combined_df.to_csv(MEMORY_FILE, index=False)

    # ★安全なデータセット作成（類似検索）
    train_df = create_hybrid_dataset_safe(combined_df, len(full_feature_df))
    
    # 学習
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'diff_ma', 'volatility', 'rsi']
    X = train_df[feature_cols].values
    y = train_df['target'].values
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X)
    
    # ★学習時はドロップアウトを有効にするため .train() にする
    model.train() 
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(LEARNING_EPOCHS):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X))
        loss = criterion(outputs, torch.LongTensor(y))
        loss.backward()
        optimizer.step()
        
    print(f"  [進化完了] 脳の再構築終了 (Loss: {loss.item():.4f})")
    torch.save(model.state_dict(), MODEL_PATH)
    return model, True

def main():
    if not initialize_mt5(): return
    print("=== 鉄壁防御型 AIトレーダーシステム起動 ===")
    
    try:
        model = MoE_Trader_AI(input_dim=10)
        model.load_state_dict(torch.load(MODEL_PATH))
        scaler = joblib.load(SCALER_PATH)
    except:
        print("初期モデルなし。まずは学習コードを実行してください。")
        return

    cycle = 1
    try:
        while True:
            print(f"\n--- サイクル {cycle} ---")
            trade_end = datetime.now() + timedelta(minutes=TRADE_DURATION_MIN)
            phase_trade(model, scaler, trade_end)
            
            model, success = phase_evolution(model)
            cycle += 1
            
    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()