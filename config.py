# config.py
# =========================================================
# システム全般の設定
# =========================================================
import logging

# ---------------------------------------------------------
# OANDA API / MT5 接続情報
# ---------------------------------------------------------
OANDA_ACCESS_TOKEN = "ここにあなたのトークン"
OANDA_ACCOUNT_ID = "ここにあなたのアカウントID"
OANDA_ENV = "practice" # 本番は "live"

MT5_LOGIN = 12345678       # MT5 ID
MT5_PASSWORD = "password"  # MT5 Password
MT5_SERVER = "OANDA-Japan MT5 Live"

# ---------------------------------------------------------
# 取引対象
# ---------------------------------------------------------
SYMBOL = "XAUUSD"       # OANDA API用
MT5_SYMBOL = "XAUUSD"   # MT5用 (業者によって "XAUUSD.oj" 等の場合あり要確認)
TIMEFRAME_M = 1         # 1分足

# ---------------------------------------------------------
# リスク管理 / 資金管理 (重要)
# ---------------------------------------------------------
RISK_PER_TRADE = 0.02      # 1回のトレードで許容する損失率 (残高の2%)
MAX_POSITIONS = 1          # 同時に持つ最大ポジション数
MAX_SPREAD_POINTS = 50     # これ以上スプレッドが開いたらエントリーしない
MAX_DAILY_LOSS_JPY = 10000 # 1日の最大損失額（これを超えたらシステム停止）

# SL/TP設定 (ATR倍率)
ATR_SL_MULTIPLIER = 2.0    # 損切り = ATR × 2.0
ATR_TP_MULTIPLIER = 3.0    # 利確   = ATR × 3.0
MIN_LOTS = 0.01

# ---------------------------------------------------------
# AI / システム設定
# ---------------------------------------------------------
ANOMALY_THRESHOLD = 3.0    # 異常検知の閾値
MEMORY_FILE = "long_term_memory.csv"
MODEL_PATH = "best_ai_model.pth"
SCALER_PATH = "scaler.pkl"

# ログ設定
LOG_FILE = "trade_log.log"