import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pk_list = ["777398","777490","777824","777838","777854","777866"]  # 対象のゲームPK
# pk_list = ["777398"] テスト用

feature_groups = [
    "play_features",
    "situation_features",
    "hit_event",
    "rbi_impact",
    "runner_status",
    "score_difference",
    "inning_phase"
]
# runner_status の one-hot を重み付きに変換する関数
def weighted_runner_status(status_dict):
    # 任意の重みを設定（例: none=0, first=1, second=2, third=3, first-second=4, ...）
    weights = {
        "none": 0,
        "first": 1,
        "second": 2,
        "third": 3,
        "first-second": 4,
        "first-third": 5,
        "second-third": 6,
        "first-second-third": 7
    }
    # 該当するステータスの重みを返す
    for key, val in status_dict.items():
        if val:
            return [weights.get(key, 0)]
    # どれも該当しない場合は0
    return [0]

# flatten_features の runner_status 部分を書き換えるためのラッパー
def flatten_features_with_weighted_runner(minute):
    play = minute["play_features"]
    situ = minute["situation_features"]
    features = []

    if do_place == "hit_event":
        for key in ["single", "double", "triple", "home_run"]:
            features.append(int(play["hit_event"].get(key, False)))
    elif do_place == "rbi_impact":
        for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
            features.append(int(play["rbi_impact"].get(key, False)))
    elif do_place == "runner_status":
        features.extend(weighted_runner_status(situ["runner_status"]))
    elif do_place == "score_difference":
        for key in [
            "minus_less_3", "minus_2", "minus_1", "tie",
            "plus_1", "plus_2", "plus_more_3"
        ]:
            features.append(int(situ["score_difference"].get(key, False)))
    elif do_place == "inning_phase":
        features.append(int(situ["inning_phase"].get("early", False)))
    elif do_place == "play_features":
        for key in ["single", "double", "triple", "home_run"]:
            features.append(int(play["hit_event"].get(key, False)))
        for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
            features.append(int(play["rbi_impact"].get(key, False)))
    elif do_place == "situation_features":
        features.extend(weighted_runner_status(situ["runner_status"]))
        for key in [
            "minus_less_3", "minus_2", "minus_1", "tie",
            "plus_1", "plus_2", "plus_more_3"
        ]:
            features.append(int(situ["score_difference"].get(key, False)))
        features.append(int(situ["inning_phase"].get("early", False)))

    return features
for gamepk in pk_list:
    # --- ファイルの読み込み ---
    with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
        molded_data = json.load(f)
    with open(f"data/processed/{gamepk}_processed_data.json", encoding="utf-8") as f:
        processed_data = json.load(f)

    minutes_data = molded_data["minutes"]

    for do_place in feature_groups:      
            # --- 特徴量抽出関数 ---
        def flatten_features(minute):
            play = minute["play_features"]
            situ = minute["situation_features"]
            features = []
            if do_place == "hit_event":
                for key in ["single", "double", "triple", "home_run"]:
                    features.append(int(play["hit_event"].get(key, False)))
            elif do_place == "rbi_impact":
                for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
                    features.append(int(play["rbi_impact"].get(key, False)))
            elif do_place == "runner_status":
                features.extend(weighted_runner_status(situ["runner_status"]))
            elif do_place == "score_difference":
                for key in [
                    "minus_less_3", "minus_2", "minus_1", "tie",
                    "plus_1", "plus_2", "plus_more_3"
                ]:
                    features.append(int(situ["score_difference"].get(key, False)))
            elif do_place == "inning_phase":
                features.append(int(situ["inning_phase"].get("early", False)))
            elif do_place == "play_features":
                for key in ["single", "double", "triple", "home_run"]:
                    features.append(int(play["hit_event"].get(key, False)))
                for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
                    features.append(int(play["rbi_impact"].get(key, False)))
            elif do_place == "situation_features":
                features.extend(weighted_runner_status(situ["runner_status"]))
                for key in [
                    "minus_less_3", "minus_2", "minus_1", "tie",
                    "plus_1", "plus_2", "plus_more_3"
                ]:
                    features.append(int(situ["score_difference"].get(key, False)))
                features.append(int(situ["inning_phase"].get("early", False)))

            return features
        
    # --- ハイライトラベル抽出関数 ---
        def is_highlight(minute):
            runner_status_weights = {
                "none": 0,
                "first": 1,
                "second": 1,
                "third": 1,
                "first-second": 1.25,
                "first-third": 1.5,
                "second-third": 1.5,
                "first-second-third": 3,
            }
            if do_place == "play_features":
                for group in ["hit_event", "rbi_impact"]:
                    if any(minute["play_features"][group].values()):
                        return 1
            elif do_place == "situation_features":
                for group in ["runner_status", "score_difference", "inning_phase"]:
                    if any(minute["situation_features"][group].values()):
                        return 1
            elif do_place == "hit_event":
                if any(minute["play_features"]["hit_event"].values()):
                    return 1
            elif do_place == "rbi_impact":
                if any(minute["play_features"]["rbi_impact"].values()):
                    return 1
            elif do_place == "runner_status":
                score = 0
                for key, weight in runner_status_weights.items():
                    if minute["situation_features"]["runner_status"].get(key, False):
                        score += weight
                return score >= 1  # この閾値は調整可能
            elif do_place == "score_difference":
                if any(minute["situation_features"]["score_difference"].values()):
                    return 1
            elif do_place == "inning_phase":
                if any(minute["situation_features"]["inning_phase"].values()):
                    return 1
            return 0

        # --- 特徴量とラベル作成 ---
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

        # --- ロジスティック回帰モデル ---
        if len(np.unique(y)) > 1:
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
        else:
            print(f"⚠️ ラベルが一種類しかないため、モデル学習不可。{do_place}")
            probs = [0.0] * len(X)

        # --- 出力データの構築 ---
        output_data = {}
        for idx, minute_str in enumerate(minute_keys):
            inning_idx = str(int(minute_str) // 60)  # 仮の分割基準
            play_idx = str(int(minute_str) % 60)
            try:
                runner_status_dict = minutes_data[minute_str][0]["situation_features"]["runner_status"]
                runner_status_true = [key for key, val in runner_status_dict.items() if val]
                runner_status_str = runner_status_true[0] if runner_status_true else None
            except (KeyError, IndexError):
                runner_status_str = {}

            try:
                event = processed_data[inning_idx][play_idx]
                output_data[minute_str] = {
                    "prob": round(probs[idx] * 100, 1),
                    "runner_status": runner_status_str
                }
            except KeyError:
                output_data[minute_str] = {
                    "prob": round(probs[idx] * 100, 1),
                    "runner_status": None
                }

        # --- JSONファイルとして保存 ---
        with open(f"data/LRA_data/LRA_{do_place}/{gamepk}_highlight_predictions_{do_place}_with_event_info.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✅ JSONファイルを出力しました: {gamepk}_highlight_predictions_{do_place}_with_event_info.json")
