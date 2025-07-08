import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
from itertools import combinations

pk_list = ["777398", "777490", "777824", "777838", "777854", "777866"]

# 組み合わせ対象（単体では使わない）
feature_groups = [
    "hit_event",
    "rbi_impact",
    "runner_status",
    "score_difference",
    "inning_phase",
]

# 組み合わせ生成
feature_combinations = list(combinations(feature_groups, 2))

for gamepk in pk_list:
    with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
        molded_data = json.load(f)
    with open(f"data/processed/{gamepk}_processed_data.json", encoding="utf-8") as f:
        processed_data = json.load(f)

    minutes_data = molded_data["minutes"]

    for fg1, fg2 in feature_combinations:
        combination_name = f"{fg1}__{fg2}"

        def flatten_features(minute):
            play = minute["play_features"]
            situ = minute["situation_features"]
            features = []

            def extract(group):
                f = []
                if group == "hit_event":
                    for key in ["single", "double", "triple", "home_run"]:
                        f.append(int(play["hit_event"].get(key, False)))
                elif group == "rbi_impact":
                    for key in [
                        "regular_rbi",
                        "tie_rbi",
                        "go_ahead_rbi",
                        "sayonara_rbi",
                    ]:
                        f.append(int(play["rbi_impact"].get(key, False)))
                elif group == "runner_status":
                    for key in [
                        "none",
                        "first",
                        "second",
                        "third",
                        "first-second",
                        "first-third",
                        "second-third",
                        "first-second-third",
                    ]:
                        f.append(int(situ["runner_status"].get(key, False)))
                elif group == "score_difference":
                    for key in [
                        "minus_less_3",
                        "minus_2",
                        "minus_1",
                        "tie",
                        "plus_1",
                        "plus_2",
                        "plus_more_3",
                    ]:
                        f.append(int(situ["score_difference"].get(key, False)))
                elif group == "inning_phase":
                    f.append(int(situ["inning_phase"].get("early", False)))
                return f

            features += extract(fg1)
            features += extract(fg2)
            return features

        def is_highlight(minute):
            play = minute["play_features"]
            situ = minute["situation_features"]

            def group_has_true(group):
                if group == "hit_event":
                    return any(play["hit_event"].values())
                elif group == "rbi_impact":
                    return any(play["rbi_impact"].values())
                elif group == "runner_status":
                    return any(situ["runner_status"].values())
                elif group == "score_difference":
                    return any(situ["score_difference"].values())
                elif group == "inning_phase":
                    return any(situ["inning_phase"].values())
                return False

            return int(group_has_true(fg1) or group_has_true(fg2))

        X, y, minute_keys = [], [], []

        for minute_str, minute_list in minutes_data.items():
            if not minute_list:
                continue
            minute = minute_list[0]
            X.append(flatten_features(minute))
            y.append(is_highlight(minute))
            minute_keys.append(minute_str)

        X = np.nan_to_num(np.array(X))
        y = np.array(y)

        if len(np.unique(y)) > 1:
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
        else:
            print(f"⚠️ ラベルが一種類のみ: {gamepk} - {combination_name}")
            probs = [0.0] * len(X)

        output_data = {}
        for idx, minute_str in enumerate(minute_keys):
            inning_idx = str(int(minute_str) // 60)
            play_idx = str(int(minute_str) % 60)
            try:
                event = processed_data[inning_idx][play_idx]
                output_data[minute_str] = {
                    "highlight_probability": round(probs[idx] * 100, 1),
                    "batter": event["batter"]["full_name"],
                    "event_type": event["event_type"],
                }
            except KeyError:
                output_data[minute_str] = {
                    "highlight_probability": round(probs[idx] * 100, 1),
                    "batter": None,
                    "event_type": None,
                }

        output_dir = f"data/LRA_combined_data/LRA_{combination_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{gamepk}_highlight_predictions_{combination_name}_with_event_info.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✅ JSONファイルを出力しました: {output_path}")
