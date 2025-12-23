import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from config import OANDA_ACCESS_TOKEN, OANDA_ACCOUNT_ID, OANDA_ENV, SYMBOL

# ==========================================
# 共通の特徴量エンジニアリング (ATR対応)
# ==========================================
def feature_engineering(df):
    df = df.copy()
    # 移動平均
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['diff_ma'] = (df['close'] - df['ma20']) / df['ma20']
    
    # ATR (ボラティリティ)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 対数収益率
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_data_and_save():
    client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENV)
    oanda_symbol = SYMBOL.replace("XAUUSD", "XAU_USD")
    print(f"--- {oanda_symbol} データ収集開始 (3クラス分類用) ---")
    
    params = {"granularity": "M1", "count": 5000, "price": "M"}

    try:
        r = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
        client.request(r)
    except Exception as e:
        print(f"API Error: {e}")
        return

    data = []
    for c in r.response['candles']:
        if c['complete']:
            data.append({
                "time": c['time'],
                "open": float(c['mid']['o']), "high": float(c['mid']['h']),
                "low": float(c['mid']['l']), "close": float(c['mid']['c']),
                "volume": int(c['volume'])
            })
    
    df = pd.DataFrame(data)
    
    # 特徴量作成
    df = feature_engineering(df)
    
    # -----------------------------------------------------
    # 【重要】3クラス分類の正解ラベル作成
    # -----------------------------------------------------
    # 未来(1分後)の価格変動を見る
    future_close = df['close'].shift(-1)
    current_close = df['close']
    
    # 閾値設定 (例: 価格の0.01%以上動いたらトレンドとみなす)
    # ゴールド2000ドルなら0.2ドル程度の動きが必要
    threshold = current_close * 0.0001
    
    conditions = [
        (future_close > current_close + threshold), # 上昇 (1: BUY)
        (future_close < current_close - threshold)  # 下落 (2: SELL)
    ]
    choices = [1, 2]
    
    # 条件に当てはまらない場合は 0 (STAY)
    df['target'] = np.select(conditions, choices, default=0)
    
    df.dropna(inplace=True)
    df.to_csv("training_data.csv", index=False)
    print(f"保存完了: training_data.csv ({len(df)}行)")
    print("ラベル分布:\n", df['target'].value_counts())

if __name__ == "__main__":
    get_data_and_save()