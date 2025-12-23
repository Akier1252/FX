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
import logging
import math

#ChtaGPTとClaudeによるアドバイス修正版（2025/12/22時点）

# 設定読み込み
from config import (
    OANDA_ACCESS_TOKEN, OANDA_ENV, SYMBOL, 
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_SYMBOL
)

# =========================================================
# 【設定】プロ運用パラメータ
# =========================================================
TRADE_DURATION_MIN = 60
LEARNING_EPOCHS = 30
MEMORY_FILE = "long_term_memory.csv"
PAST_SAMPLE_RATIO = 0.5

# リスク管理
RISK_PER_TRADE = 0.02      # 1トレードあたりの許容リスク（口座残高の2%）
MAX_POSITIONS = 1          # 同時保有最大ポジション数
ATR_SL_MULTIPLIER = 2.0    # 損切り幅 = ATR × 2.0
ATR_TP_MULTIPLIER = 3.0    # 利確幅 = ATR × 3.0
MIN_LOTS = 0.01            # 最小ロット

# 安全装置
ANOMALY_THRESHOLD = 3.0    # マハラノビス距離閾値
MAX_SPREAD_POINTS = 50     # 許容スプレッド

MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

# ログ設定
logging.basicConfig(
    filename='trade_log.log', 
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# =========================================================
# AIモデル（3クラス分類: 0=STAY, 1=BUY, 2=SELL）
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
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_experts=5):
        # output_dim=3 に変更 (Stay/Buy/Sell)
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
# 特徴量エンジニアリング（ATR必須）
# =========================================================
def feature_engineering(df):
    df = df.copy()
    
    # 基本指標
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma20'] = df['ma20'].replace(0, np.nan)
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    
    # ATR (Average True Range) - ボラティリティ測定用
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
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

FEATURE_COLS = ['open', 'high', 'low', 'close', 'tick_volume', 'ma5', 'ma20', 'diff_ma', 'atr', 'rsi']

# =========================================================
# リスク管理・資金管理システム
# =========================================================
def check_positions():
    """現在のポジション数を確認"""
    positions = mt5.positions_get(symbol=MT5_SYMBOL)
    if positions is None: return 0
    return len(positions)

def calculate_dynamic_lot(risk_pips):
    """
    口座残高と許容リスクからロット数を計算
    (残高 * 2%) / (損切り幅 * 1pipsの価値)
    """
    account_info = mt5.account_info()
    if account_info is None: return MIN_LOTS
    
    balance = account_info.balance
    risk_amount = balance * RISK_PER_TRADE # 許容損失額（円）
    
    # 損切り幅(point)をpips換算等は業者によるが、
    # 簡易的に: ロット = リスク額 / (変動幅 * 契約サイズ)
    # ※MT5の通貨ごとのTickValue取得が必要
    symbol_info = mt5.symbol_info(MT5_SYMBOL)
    if symbol_info is None: return MIN_LOTS
    
    # XAUUSDの場合、1lot=100ozなど。
    # 簡易計算: 損切り幅(価格差) * 契約サイズ * レート
    # ここでは1pips(または1ドル)動いた時の損益(tick_value)を使う
    
    # リスク幅(価格)
    risk_price_dist = risk_pips * symbol_info.point
    
    # 1ロットでその幅動いた時の損失
    # profit_modeなどにより計算異なるが、tick_valueを利用して近似
    # tick_value = 1point動いた時の1lotあたりの評価額
    loss_per_lot = risk_pips * symbol_info.trade_tick_value
    
    if loss_per_lot == 0: return MIN_LOTS
    
    lots = risk_amount / loss_per_lot
    
    # 丸め処理
    lots = max(MIN_LOTS, round(lots, 2))
    return lots

# =========================================================
# 実行ロジック（ショート対応）
# =========================================================
def execute_trade_pro(action, confidence, current_atr, current_price):
    # 1. ポジション過多ならエントリーしない
    if check_positions() >= MAX_POSITIONS:
        print("  [Skip] ポジション上限到達")
        return

    # 2. スプレッドチェック
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    spread = (tick.ask - tick.bid) / mt5.symbol_info(MT5_SYMBOL).point
    if spread > MAX_SPREAD_POINTS:
        print(f"  [Skip] スプレッド拡大: {spread}")
        return

    # 3. ATRに基づく動的SL/TP計算
    # ATRは価格幅そのもの（例：ゴールドで2.5ドル）
    sl_dist = current_atr * ATR_SL_MULTIPLIER
    tp_dist = current_atr * ATR_TP_MULTIPLIER
    
    # Point単位に変換（注文用）
    point = mt5.symbol_info(MT5_SYMBOL).point
    sl_points = sl_dist / point
    tp_points = tp_dist / point

    # 4. ロット計算（SL幅に基づく）
    lots = calculate_dynamic_lot(sl_points)

    # 5. 注文パラメータ作成
    if action == 1: # BUY
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_dist
        tp = price + tp_dist
        color = "Blue"
    elif action == 2: # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_dist
        tp = price - tp_dist
        color = "Red"
    else:
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": f"AI_v11_{color}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # 6. 送信 & エラーチェック
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"注文失敗: {result.comment}")
        print(f"  ❌ 注文失敗: {result.comment}")
    else:
        log_msg = f"{'BUY' if action==1 else 'SELL'} Lot:{lots} Price:{price} SL:{sl:.2f} TP:{tp:.2f} (ATR:{current_atr:.2f})"
        logging.info(log_msg)
        print(f"  ✅ {log_msg}")

# =========================================================
# メイン処理（簡略化のため一部関数は省略、既存コード参照）
# =========================================================
# ... (get_mt5_features, check_anomaly_mahalanobis, phase_evolution などは前回と同じ) ...
# ※ただし、get_mt5_features は ATR を返すように修正が必要 ↓

def get_mt5_features_pro():
    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M1, 0, 100)
    if rates is None or len(rates) == 0: return None, None, None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = feature_engineering(df)
    latest = df.iloc[-1]
    
    # 特徴量配列
    features = [latest[col] for col in FEATURE_COLS]
    
    # ATRと現在価格も返す（資金管理用）
    return np.array(features).reshape(1, -1), latest['atr'], latest['close']

def main():
    if not mt5.initialize(): return
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER): return
    print("=== 実戦運用・完成形 (ATR資金管理/ショート対応) ===")

    try:
        # 出力次元が3になったのでモデル再構築必須
        model = MoE_Trader_AI(input_dim=len(FEATURE_COLS), output_dim=3) 
        if os.path.exists(MODEL_PATH):
            # ※注意: 次元数が変わったので古いモデルは読み込めない可能性大
            # その場合は再学習が必要
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
            except:
                print("⚠️ モデル構造が変わりました。再学習が必要です。")
        
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        memory_df = pd.read_csv(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else None
        
    except Exception as e:
        print(f"初期化エラー: {e}")
        return

    cycle = 1
    try:
        while True:
            print(f"\n--- サイクル {cycle} ---")
            end_time = datetime.now() + timedelta(minutes=TRADE_DURATION_MIN)
            
            while datetime.now() < end_time:
                try:
                    features, current_atr, current_price = get_mt5_features_pro()
                    if features is not None and scaler is not None:
                        # 異常検知
                        # ... (前回と同じマハラノビス・チェック) ...

                        # 予測
                        feat_scaled = scaler.transform(features)
                        model.eval()
                        with torch.no_grad():
                            output = model(torch.FloatTensor(feat_scaled))
                            probs = torch.nn.functional.softmax(output, dim=1)
                            action = torch.argmax(probs, dim=1).item()
                            conf = probs[0][action].item()
                        
                        action_str = ["STAY", "BUY", "SELL"][action]
                        print(f"\r  AI: {action_str} ({conf:.1%}) ATR:{current_atr:.2f}", end="")

                        if action != 0 and conf > 0.8:
                            print(f"\n  ★ {action_str} シグナル点灯！")
                            execute_trade_pro(action, conf, current_atr, current_price)
                            time.sleep(60)

                    time.sleep(1)
                except Exception as e:
                    print(e)
                    time.sleep(5)
            
            # 進化フェーズ (前回と同じだが、targetの作り方を3値分類に変える必要あり)
            # ...
            cycle += 1
            
    except KeyboardInterrupt:
        mt5.shutdown()

if __name__ == "__main__":
    main()