import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pk_list = ["777398","777490","777824","777838","777854","777866"]  # 対象のゲームPK

for gamepk in pk_list:
    # --- ファイルの読み込み ---
    with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
        molded_data = json.load(f)
    with open(f"data/processed/{gamepk}_processed_data.json", encoding="utf-8") as f:
        processed_data = json.load(f)

    minutes_data = molded_data["minutes"]

    # --- 特徴量抽出関数 ---
    def flatten_features(minute):
        play = minute["play_features"]
        situ = minute["situation_features"]
        features = []

        for key in ["single", "double", "triple", "home_run"]:
            features.append(int(play["hit_event"].get(key, False)))
        for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
            features.append(int(play["rbi_impact"].get(key, False)))
        for key in [
            "none", "first", "second", "third", "first-second",
            "first-third", "second-third", "first-second-third"
        ]:
            features.append(int(situ["runner_status"].get(key, False)))
        features.append(int(situ.get("is_goahead_runner_on_base", False)))
        for key in [
            "minus_less_3", "minus_2", "minus_1", "tie",
            "plus_1", "plus_2", "plus_more_3"
        ]:
            features.append(int(situ["score_difference"].get(key, False)))
        features.append(int(situ["inning_phase"].get("early", False)))
        return features

    # --- ハイライトラベル抽出関数 ---
    def is_play_feature_highlight(minute):
        for group in ["hit_event", "rbi_impact"]:
            if any(minute["play_features"][group].values()):
                return 1
        return 0

    # --- 特徴量とラベル作成 ---
    X, y, minute_keys = [], [], []

    for minute_str, minute_list in minutes_data.items():
        if not minute_list:
            continue
        minute = minute_list[0]
        X.append(flatten_features(minute))
        y.append(is_play_feature_highlight(minute))
        minute_keys.append(minute_str)

    X = np.nan_to_num(np.array(X))
    y = np.array(y)

    # --- ロジスティック回帰モデル ---
    if len(np.unique(y)) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
    else:
        print("⚠️ ラベルが一種類しかないため、モデル学習不可。")
        probs = [0.0] * len(X)

    # --- 出力データの構築 ---
    output_data = {}
    for idx, minute_str in enumerate(minute_keys):
        inning_idx = str(int(minute_str) // 60)  # 仮の分割基準
        play_idx = str(int(minute_str) % 60)
        try:
            event = processed_data[inning_idx][play_idx]
            output_data[minute_str] = {
                "highlight_probability": round(probs[idx], 3),
                "batter": event["batter"]["full_name"],
                "event_type": event["event_type"]
            }
        except KeyError:
            output_data[minute_str] = {
                "highlight_probability": round(probs[idx], 3),
                "batter": None,
                "event_type": None
            }

    # --- JSONファイルとして保存 ---
    with open(f"data/LRA_data/{gamepk}_highlight_predictions_with_event_info.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSONファイルを出力しました: {gamepk}_highlight_predictions_with_event_info.json")
