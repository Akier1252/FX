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
from datetime import datetime, timedelta, timezone
import os
import logging

#ChtaGPTによるアドバイス修正版（2025/12/22時点）

# 設定読み込み
from config import (
    OANDA_ACCESS_TOKEN, OANDA_ENV, SYMBOL, 
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL
)

# =========================================================
# 【設定】実戦強化パラメータ
# =========================================================
TRADE_DURATION_MIN = 60
LEARNING_EPOCHS = 30
MEMORY_FILE = "long_term_memory.csv"
PAST_SAMPLE_RATIO = 0.5
LOTS = 0.01

# --- リスク管理（キルスイッチ） ---
MAX_DAILY_LOSS_JPY = 5000  # 1日の許容損失額（円）
MAX_SPREAD_POINTS = 50     # 許容最大スプレッド（これ以上開いたらエントリーしない）

# --- 安全装置 ---
ANOMALY_THRESHOLD = 3.0    # マハラノビス距離の閾値（これを超えたら異常）

# --- ファイルパス ---
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

# ログ設定
logging.basicConfig(
    filename='trade_log.log', 
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# =========================================================
# AIモデル（MC Dropout対応）
# =========================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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
            nn.Dropout(p=0.2),
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
# 【強化点1】高度な特徴量エンジニアリング
# =========================================================
def feature_engineering(df):
    df = df.copy()
    
    # 既存指標
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma20'] = df['ma20'].replace(0, np.nan)
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    
    # ATR (True Rangeの平均) - ボラティリティのより正確な指標
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 対数収益率（価格そのものではなく変化率を見るため重要）
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

# 特徴量のカラム定義（ATRとLogRetを追加）
FEATURE_COLS = ['open', 'high', 'low', 'close', 'tick_volume', 'ma5', 'ma20', 'diff_ma', 'atr', 'log_ret', 'rsi']

# =========================================================
# 【強化点2】マハラノビス距離による異常検知
# =========================================================
def calculate_mahalanobis(x, data):
    """
    x: 現在のベクトル (1, feature_dim)
    data: 過去データの分布 (N, feature_dim)
    """
    # 共分散行列の計算
    cov = np.cov(data, rowvar=False)
    # 逆行列（擬似逆行列）
    try:
        inv_cov = np.linalg.pinv(cov)
    except:
        return 0.0 # 計算不可時は0
        
    mu = np.mean(data, axis=0)
    diff = x - mu
    # 距離計算
    dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return dist.item()

def check_anomaly_mahalanobis(current_features, memory_df, scaler):
    """
    単純な閾値ではなく、多変量解析で「異常」を検知する
    """
    if memory_df is None or len(memory_df) < 50:
        return False, 0.0

    # 学習済みスケーラーで正規化
    # ※メモリ内のデータも正規化する必要があるため、計算コスト削減のため
    #   直近N件だけサンプリングして分布を作るのが実用的
    sample_memory = memory_df.iloc[-200:][FEATURE_COLS].values
    sample_memory_scaled = scaler.transform(sample_memory)
    
    current_scaled = scaler.transform(current_features)
    
    dist = calculate_mahalanobis(current_scaled, sample_memory_scaled)
    
    if dist > ANOMALY_THRESHOLD:
        return True, dist
    return False, dist

# =========================================================
# 【強化点3】MC Dropoutによる不確実性推定
# =========================================================
def predict_with_uncertainty(model, x_tensor, n_samples=20):
    """
    推論時にもDropoutを有効にして複数回予測し、
    AIの「迷い（分散）」を計測する
    """
    model.train() # Dropoutを有効にする
    probs_list = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            probs_list.append(probs.numpy())
            
    probs_array = np.array(probs_list) # shape: (n_samples, batch, classes)
    
    # 平均予測確率
    mean_probs = probs_array.mean(axis=0)
    # 予測のばらつき（標準偏差）＝ 不確実性
    uncertainty = probs_array.std(axis=0).max(axis=1) # 最も高い分散を採用
    
    action = np.argmax(mean_probs, axis=1)[0]
    confidence = mean_probs[0][action]
    uncertainty_score = uncertainty[0]
    
    return action, confidence, uncertainty_score

# =========================================================
# 【強化点4】リスク管理と実行（キルスイッチ・エラー処理）
# =========================================================
def check_account_safety():
    """口座状況を確認し、危険ならFalseを返す（キルスイッチ）"""
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("口座情報が取得できません")
        return False

    # 1. 今日の損益チェック
    # ※MT5で「今日の損益」を正確に取るには履歴取得が必要だが、
    # 簡易的に equity - balance で含み損を見る、もしくは別ロジックが必要。
    # ここでは「有効証拠金が残高を大きく下回っている場合」を緊急事態とする
    drawdown = account_info.balance - account_info.equity
    if drawdown > MAX_DAILY_LOSS_JPY:
        logging.critical(f"【緊急停止】ドローダウンが許容値を超えました: {drawdown}")
        print(f"\n★ KILL SWITCH ACTIVATED: Drawdown {drawdown} > {MAX_DAILY_LOSS_JPY}")
        return False
        
    return True

def execute_trade_robust(action, confidence, uncertainty):
    if action == 0: return 

    # 1. スプレッドチェック
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if tick is None: return
    
    spread = (tick.ask - tick.bid) / mt5.symbol_info(MT5_SYMBOL).point
    if spread > MAX_SPREAD_POINTS:
        print(f"  [Skip] スプレッド拡大中: {spread}")
        return

    # 2. 不確実性チェック（AIが迷っている時は入らない）
    if uncertainty > 0.1: # 閾値は調整が必要
        print(f"  [Skip] AIの不確実性が高い: {uncertainty:.3f}")
        return

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
        "magic": 999999,
        "comment": f"AI_Conf:{confidence:.2f}_Unc:{uncertainty:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # 3. 注文送信とエラーハンドリング
    result = mt5.order_send(request)
    
    if result is None:
        logging.error("order_send returned None (Timeout?)")
        print("  ❌ 注文タイムアウト")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"注文失敗: {result.retcode} - {result.comment}")
        print(f"  ❌ 注文失敗: {result.comment}")
    else:
        logging.info(f"注文成功: Ticket {result.order}")
        print(f"  ✅ 注文成功 (Ticket: {result.order})")

# =========================================================
# データ取得・学習関連（統合版）
# =========================================================
def get_mt5_features():
    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M1, 0, 100) # ATR計算のため少し多めに
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = feature_engineering(df)
    latest = df.iloc[-1]
    
    # カラム順序を厳守して配列化
    features = [latest[col] for col in FEATURE_COLS]
    return np.array(features).reshape(1, -1)

def create_hybrid_dataset_safe(full_memory_df, recent_df_len):
    # 類似検索ロジック（前回のものを使用）
    # ※Feature Colsが増えたので計算対象も増やす
    calc_cols = ['ma5', 'ma20', 'diff_ma', 'atr', 'rsi'] # 類似度計算に使う主要指標
    
    if len(full_memory_df) < recent_df_len * 2:
        return full_memory_df.iloc[-recent_df_len:]

    current_state = full_memory_df.iloc[-1][calc_cols].values.astype(float)
    past_memory = full_memory_df.iloc[:-recent_df_len].copy()
    
    # 距離計算
    past_features = past_memory[calc_cols].values.astype(float)
    distances = np.sum((past_features - current_state) ** 2, axis=1)
    past_memory['distance'] = distances
    past_memory.sort_values(by='distance', ascending=True, inplace=True)
    
    # 上位30%からサンプリング
    sample_size = int(recent_df_len * (PAST_SAMPLE_RATIO / (1 - PAST_SAMPLE_RATIO)))
    candidate_pool = past_memory.iloc[: int(len(past_memory) * 0.3)]
    
    if len(candidate_pool) >= sample_size:
        past_sample = candidate_pool.sample(n=sample_size)
    else:
        past_sample = candidate_pool

    recent_data = full_memory_df.iloc[-recent_df_len:]
    return pd.concat([recent_data, past_sample]).sample(frac=1).reset_index(drop=True)

def phase_evolution(model):
    print("\n\n[進化モード] データ収集と学習...")
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENV)
    params = {"granularity": "M1", "count": 1000, "price": "M"}
    try:
        r = instruments.InstrumentsCandles(instrument=SYMBOL.replace("XAUUSD", "XAU_USD"), params=params)
        client.request(r)
    except Exception as e:
        logging.error(f"OANDA API Error: {e}")
        return model, False

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

    # アトミックな書き込み（データ破損防止）
    if os.path.exists(MEMORY_FILE):
        old_df = pd.read_csv(MEMORY_FILE)
        combined_df = pd.concat([old_df, full_feature_df])
        combined_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    else:
        combined_df = full_feature_df
    
    # 一時ファイルに書いてからリネーム
    tmp_file = MEMORY_FILE + ".tmp"
    combined_df.to_csv(tmp_file, index=False)
    os.replace(tmp_file, MEMORY_FILE)

    train_df = create_hybrid_dataset_safe(combined_df, len(full_feature_df))
    
    # 学習
    X = train_df[FEATURE_COLS].values
    y = train_df['target'].values
    
    scaler = joblib.load(SCALER_PATH)
    # スケーラーのアップデート（新データに追従するため）
    scaler.partial_fit(X) 
    joblib.dump(scaler, SCALER_PATH) # 保存
    
    X_scaled = scaler.transform(X)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(LEARNING_EPOCHS):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_scaled))
        loss = criterion(outputs, torch.LongTensor(y))
        loss.backward()
        optimizer.step()
        
    print(f"  [完了] 進化完了 Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    return model, True

def initialize_mt5():
    if not mt5.initialize(): return False
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER): return False
    return True

# =========================================================
# メインループ
# =========================================================
def main():
    if not initialize_mt5(): return
    print("=== 実戦強化型（堅牢版）AIトレーダー起動 ===")
    
    try:
        # FEATURE_COLSの次元数に合わせてモデル初期化
        model = MoE_Trader_AI(input_dim=len(FEATURE_COLS))
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
        else:
            print("初期モデルがありません。学習コードを先に実行してください。")
            return
            
        scaler = joblib.load(SCALER_PATH)
        
        # メモリ読み込み（異常検知用）
        memory_df = pd.read_csv(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else None
        
    except Exception as e:
        print(f"初期化エラー: {e}")
        return

    cycle = 1
    try:
        while True:
            # キルスイッチ確認
            if not check_account_safety():
                print("システムを停止します。")
                break

            print(f"\n--- サイクル {cycle} ---")
            end_time = datetime.now() + timedelta(minutes=TRADE_DURATION_MIN)
            
            # --- 戦闘フェーズ ---
            while datetime.now() < end_time:
                try:
                    features = get_mt5_features()
                    if features is not None:
                        # 1. マハラノビス異常検知
                        is_anomaly, dist = check_anomaly_mahalanobis(features, memory_df, scaler)
                        if is_anomaly:
                            print(f"\r  ⚠️ 異常検知 (Dist:{dist:.1f}) -> 待機", end="")
                            time.sleep(5)
                            continue

                        # 2. 不確実性付き予測
                        feat_scaled = scaler.transform(features)
                        x_tensor = torch.FloatTensor(feat_scaled)
                        
                        action, conf, unc = predict_with_uncertainty(model, x_tensor)
                        
                        print(f"\r  AI: {'BUY' if action==1 else 'STAY'} (Conf:{conf:.0%} Unc:{unc:.2f})", end="")
                        
                        # 3. エントリー判定（確信度が高く、迷いが少ない時）
                        if action == 1 and conf > 0.85:
                            print("\n  ★ 堅牢なチャンス検知！")
                            execute_trade_robust(action, conf, unc)
                            time.sleep(60)
                            
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Trade Loop Error: {e}")
                    time.sleep(5)
            
            # --- 進化フェーズ ---
            model, success = phase_evolution(model)
            if success and os.path.exists(MEMORY_FILE):
                memory_df = pd.read_csv(MEMORY_FILE) # メモリリフレッシュ
                
            cycle += 1
            
    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()