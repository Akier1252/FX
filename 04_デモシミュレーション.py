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
# AIモデル定義（読み込み用）
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

# --- シミュレーション設定 ---
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"
INITIAL_BALANCE = 10000  # スタート資金: 1万円
LOTS = 0.01              # 取引ロット数
SPREAD_COST = 20         # 仮想スプレッドコスト（ポイント単位）

# グローバル変数（シミュレーション状態管理）
current_balance = INITIAL_BALANCE
position = None  # 現在のポジション {"price": entry_price, "sl": stop_loss, "tp": take_profit}

def initialize_mt5():
    if not mt5.initialize():
        print("MT5初期化失敗")
        return False
    mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    print(">>> デモシミュレーションモード起動 <<<")
    print(f"初期資金: {INITIAL_BALANCE}円")
    return True

def get_latest_features():
    """本番と同じロジックで特徴量を作成"""
    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5.TIMEFRAME_M1, 0, 50)
    if rates is None or len(rates) == 0: return None
    
    df = pd.DataFrame(rates)
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
    
    latest = df.iloc[-1]
    features = [
        latest['open'], latest['high'], latest['low'], latest['close'], latest['tick_volume'],
        latest['ma5'], latest['ma20'], latest['diff_ma'], latest['volatility'], latest['rsi']
    ]
    return np.array(features).reshape(1, -1)

def virtual_trade_logic(action, current_price, point):
    global current_balance, position
    
    # ポジション保有中の管理（利確・損切り判定）
    if position is not None:
        # 利益計算 (Goldなどは 1lot=100oz などの計算が必要だが、簡易的に差額で計算)
        # 0.01ロット * 変動幅(point)
        # MT5の損益計算は複雑なため、ここでは簡易シミュレーションとして
        # 「価格差 × 100円」程度の変動イメージで計算します
        
        diff = current_price - position['price']
        
        # 利確 (TP) 到達？
        if current_price >= position['tp']:
            profit = (position['tp'] - position['price']) * 100 * LOTS # 簡易計算
            current_balance += profit
            print(f"\n[決済] 利確成功！ (+{int(profit)}円) 残高: {int(current_balance)}円")
            position = None
            
        # 損切り (SL) 到達？
        elif current_price <= position['sl']:
            loss = (position['price'] - position['sl']) * 100 * LOTS
            current_balance -= loss
            print(f"\n[決済] 損切り... (-{int(loss)}円) 残高: {int(current_balance)}円")
            position = None
            
        return # ポジションがある時は新規エントリーしない

    # 新規エントリー判断
    if action == 1: # AIが「買い」と判断
        sl = current_price - 500 * point # 損切り幅
        tp = current_price + 1000 * point # 利確幅
        
        # スプレッドコストを引いて仮想エントリー
        entry_price = current_price + (SPREAD_COST * point)
        
        position = {
            "price": entry_price,
            "sl": sl,
            "tp": tp
        }
        print(f"\n[新規] 仮想買いエントリー @ {entry_price:.2f}")
        print(f"       目標: {tp:.2f} / 撤退: {sl:.2f}")

def main():
    if not initialize_mt5(): return

    # モデルロード
    try:
        model = MoE_Trader_AI(input_dim=10)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        scaler = joblib.load(SCALER_PATH)
    except:
        print("モデルファイルが見つかりません。学習を実行してください。")
        return

    print("--- 1万円チャレンジ（シミュレーション）開始 ---")
    
    try:
        while True:
            # 現在価格取得
            tick = mt5.symbol_info_tick(MT5_SYMBOL)
            if tick is None: continue
            current_price = tick.ask
            point = mt5.symbol_info(MT5_SYMBOL).point

            # AI予測
            features = get_latest_features()
            if features is not None:
                feat_scaled = scaler.transform(features)
                with torch.no_grad():
                    output = model(torch.FloatTensor(feat_scaled))
                    probs = torch.nn.functional.softmax(output, dim=1)
                    action = torch.argmax(probs, dim=1).item()
                    conf = probs[0][action].item()

                # ログ表示 (保有中は損益目安も表示)
                status = "待機中"
                if position:
                    diff = current_price - position['price']
                    pnl_est = diff * 100 * LOTS
                    status = f"保有中 (含み益: {int(pnl_est)}円)"
                
                print(f"\r価格:{current_price:.2f} | AI:{'BUY' if action==1 else 'STAY'}({conf:.0%}) | 資金:{int(current_balance)}円 | {status}", end="")

                # エントリー判定（確信度80%以上）
                if action == 1 and conf > 0.8:
                    virtual_trade_logic(1, current_price, point)
                
                # 保有中の決済判定呼び出し
                if position:
                    virtual_trade_logic(0, current_price, point)

            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\n--- シミュレーション終了 ---")
        print(f"最終資金: {int(current_balance)}円")
        print(f"収支: {'+' if current_balance >= INITIAL_BALANCE else ''}{int(current_balance - INITIAL_BALANCE)}円")
        mt5.shutdown()

if __name__ == "__main__":
    main()