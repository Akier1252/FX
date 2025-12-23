# =========================================================
# 【新機能】類似検索によるスマート復習システム
# =========================================================
def create_hybrid_dataset(full_memory_df, recent_df_len):
    """
    【改良版】
    ランダムではなく、「現在の相場環境に似ている過去データ」を
    優先的に抽出して復習データセットを作る。
    """
    # 1. 現在の相場の特徴（直近の最後の行）を取得
    # ※学習に使わないカラム（time, targetなど）は除外して計算する
    feature_cols = ['ma5', 'ma20', 'diff_ma', 'volatility', 'rsi', 'volume']
    
    # データが足りない場合はそのまま返す
    if len(full_memory_df) < recent_df_len * 2:
        return full_memory_df.iloc[-recent_df_len:]

    # 現在の状況（ベクトル）
    current_state = full_memory_df.iloc[-1][feature_cols].values
    current_state = current_state.astype(float) # 計算用にfloat変換

    # 2. 過去データ（直近以外）
    past_memory = full_memory_df.iloc[:-recent_df_len].copy()
    past_features = past_memory[feature_cols].values.astype(float)

    # -----------------------------------------------------
    # 類似度計算（ユークリッド距離）
    # √((x1-x2)^2 + (y1-y2)^2 ...)
    # -----------------------------------------------------
    # ※注意: 特徴量の桁が違う（Volumeは大きくRSIは小さい）と計算が狂うため
    # 簡易的に正規化して距離を測る必要がありますが、
    # ここでは処理速度優先で「差の二乗和」で近似します。
    
    # 全過去データと現在との「距離」を一括計算
    distances = np.sum((past_features - current_state) ** 2, axis=1)
    
    # 距離をデータフレームに追加
    past_memory['distance'] = distances
    
    # 3. 距離が近い順（似ている順）にソート
    past_memory.sort_values(by='distance', ascending=True, inplace=True)
    
    # 4. 「似ているデータ」を優先抽出
    # 学習データの30%を過去データにする
    sample_size = int(recent_df_len * (PAST_SAMPLE_RATIO / (1 - PAST_SAMPLE_RATIO)))
    
    # 上位 N件（最も似ている場面）を取得
    # ★ポイント: 全てを「似ている順」にすると視野が狭くなるので
    # 「似ている上位50%」の中からランダムに選ぶとバランスが良い
    candidate_pool = past_memory.iloc[: int(len(past_memory) * 0.1)] # 上位10%に絞る
    
    if len(candidate_pool) >= sample_size:
        past_sample = candidate_pool.sample(n=sample_size)
    else:
        past_sample = candidate_pool # 足りなければあるだけ使う

    # 5. 直近データと結合
    recent_data = full_memory_df.iloc[-recent_df_len:]
    training_data = pd.concat([recent_data, past_sample]).sample(frac=1).reset_index(drop=True)
    
    print(f"  [検索] 現在(RSI:{current_state[4]:.1f})に似た過去データ {len(past_sample)}件 を抽出しました。")
    return training_data