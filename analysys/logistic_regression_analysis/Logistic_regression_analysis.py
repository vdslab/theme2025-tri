import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pk_list = ["777398","777490","777824","777838","777854","777866",]

# gamepk = pk_list[3]  # 対象のゲームPK

for pk in pk_list:
# --- データの読み込み ---
    with open(f"data/molded_data/{pk}_molded_data.json", encoding="utf-8") as f:
        molded_data = json.load(f)

    minutes_data = molded_data["minutes"]

    # --- 特徴量抽出関数 ---
    def flatten_features(minute):
        play = minute["play_features"]
        situ = minute["situation_features"]
        features = []

        # hit_event
        for key in ["single", "double", "triple", "home_run"]:
            features.append(int(play["hit_event"].get(key, False)))
        # rbi_impact
        for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
            features.append(int(play["rbi_impact"].get(key, False)))
        # runner_status
        for key in [
            "none", "first", "second", "third", "first-second",
            "first-third", "second-third", "first-second-third"
        ]:
            features.append(int(situ["runner_status"].get(key, False)))
        # is_goahead_runner_on_base
        features.append(int(situ.get("is_goahead_runner_on_base", False)))
        # score_difference
        for key in [
            "minus_less_3", "minus_2", "minus_1", "tie", 
            "plus_1", "plus_2", "plus_more_3"
        ]:
            features.append(int(situ["score_difference"].get(key, False)))
        # inning_phase
        features.append(int(situ["inning_phase"].get("early", False)))

        return features

    # --- ハイライト判定（play_features内のどれか一つでもTrueなら1） ---
    def is_play_feature_highlight(minute):
        for group in ["hit_event", "rbi_impact"]:
            if any(minute["play_features"][group].values()):
                return 1
        return 0

    # --- 特徴量とラベルの構築 ---
    X = []
    y = []

    for minute_id, minute_list in minutes_data.items():
        if not minute_list:
            continue
        minute = minute_list[0]  # 1分ごとに1イベントのみと仮定

        X.append(flatten_features(minute))
        y.append(is_play_feature_highlight(minute))

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0)  # NaN を 0 に置き換え
    y = np.array(y)

    print(f"Number of samples: {len(X)}")
    print(f"Shape of each sample: {X.shape[1]}")
    print(f"Label distribution: {np.bincount(y)}")

    # --- モデル学習と評価 ---
    if len(np.unique(y)) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        print("\n--- Classification Report ---")
        print(classification_report(y, y_pred))

        probs = model.predict_proba(X)[:, 1]
        print("\n--- Highlight Probabilities ---")
        print(probs)
    else:
        print("⚠️ Warning: ラベルが一種類のみのため、学習不可。")
