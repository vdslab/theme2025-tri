import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# pk_list = ["777398", "777490", "777824", "777838", "777854", "777866"]
pk_list = ["778199"]  # テスト用
feature_groups = [
    "play_features",
    "situation_features",
    "hit_event",
    "rbi_impact",
    "runner_status",
    "score_difference",
    "inning_phase",
]

# --- アノテーションを一括で読み込んで転置 ---
annotation_df = pd.read_csv("data/anotation_data/cluster_3.csv", dtype={"gamePK": str})
annotation_df = annotation_df.set_index("gamePK").T
annotation_df.index = annotation_df.index.map(
    lambda x: f"{int(x):03d}"
)  # index: "000", "001", ...
print(annotation_df.index)

for gamepk in pk_list:
    print(f"🧪 処理中: {gamepk}")

    # --- この試合のアノテーション列だけ抽出 ---
    if gamepk not in annotation_df.index:
        print(f"⚠️ アノテーションが存在しない: {gamepk}")
        continue

    annotation_series = annotation_df.loc[gamepk].dropna()
    annotation_dict = annotation_series.astype(
        int
    ).to_dict()  # {"000": 1, "001": 0, ...}

    # --- 試合データの読み込み ---
    try:
        with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
            molded_data = json.load(f)
        with open(
            f"data/processed/{gamepk}_processed_data.json", encoding="utf-8"
        ) as f:
            processed_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 試合データが見つかりません: {gamepk}")
        continue

    minutes_data = molded_data["minutes"]

    for do_place in feature_groups:
        print(f"▶ 特徴群: {do_place}")

        # --- flatten_features 関数定義 ---
        def flatten_features(minute):
            play = minute["play_features"]
            situ = minute["situation_features"]
            features = []
            names = []

            if do_place == "hit_event":
                for key in ["single", "double", "triple", "home_run"]:
                    features.append(int(play["hit_event"].get(key, False)))
                    names.append(f"hit_event_{key}")
            elif do_place == "rbi_impact":
                for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
                    features.append(int(play["rbi_impact"].get(key, False)))
                    names.append(f"rbi_impact_{key}")
            elif do_place == "runner_status":
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
                    features.append(int(situ["runner_status"].get(key, False)))
                    names.append(f"runner_status_{key}")
            elif do_place == "score_difference":
                for key in [
                    "minus_less_3",
                    "minus_2",
                    "minus_1",
                    "tie",
                    "plus_1",
                    "plus_2",
                    "plus_more_3",
                ]:
                    features.append(int(situ["score_difference"].get(key, False)))
                    names.append(f"score_difference_{key}")
            elif do_place == "inning_phase":
                features.append(int(situ["inning_phase"].get("early", False)))
                names.append("inning_phase_early")
            elif do_place == "play_features":
                for key in ["single", "double", "triple", "home_run"]:
                    features.append(int(play["hit_event"].get(key, False)))
                    names.append(f"hit_event_{key}")
                for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
                    features.append(int(play["rbi_impact"].get(key, False)))
                    names.append(f"rbi_impact_{key}")
            elif do_place == "situation_features":
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
                    features.append(int(situ["runner_status"].get(key, False)))
                    names.append(f"runner_status_{key}")
                for key in [
                    "minus_less_3",
                    "minus_2",
                    "minus_1",
                    "tie",
                    "plus_1",
                    "plus_2",
                    "plus_more_3",
                ]:
                    features.append(int(situ["score_difference"].get(key, False)))
                    names.append(f"score_difference_{key}")
                features.append(int(situ["inning_phase"].get("early", False)))
                names.append("inning_phase_early")

            return features, names

        # --- 特徴量とアノテーションラベルの構築 ---
        X, y, minute_keys = [], [], []
        feature_names = None  # 初期化

        for minute_str, minute_list in minutes_data.items():
            if not minute_list or minute_str not in annotation_dict:
                continue
            minute = minute_list[0]
            features, names = flatten_features(minute)
            X.append(features)
            y.append(annotation_dict[minute_str])
            minute_keys.append(minute_str)
            if feature_names is None:
                feature_names = names  # 一度だけ取得

        if not X:
            print(f"⚠️ 有効なデータが存在しない: {gamepk}, {do_place}")
            continue

        X = np.nan_to_num(np.array(X))
        y = np.array(y)

        # --- モデル学習と予測 ---
        if len(np.unique(y)) > 1:
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            weights = model.coef_[0]
        else:
            print(f"⚠️ ラベルが一種類しかないためモデル学習不可: {do_place}")
            probs = [0.0] * len(X)
            weights = []

        # --- 出力データの構築 ---
        output_data = {}
        for idx, minute_str in enumerate(minute_keys):
            inning_idx = str(int(minute_str) // 60)
            play_idx = str(int(minute_str) % 60)

            try:
                runner_status_dict = minutes_data[minute_str][0]["situation_features"][
                    "runner_status"
                ]
                runner_status_true = [
                    key for key, val in runner_status_dict.items() if val
                ]
                runner_status_str = (
                    runner_status_true[0] if runner_status_true else None
                )
            except (KeyError, IndexError):
                runner_status_str = None

            try:
                event = processed_data[inning_idx][play_idx]
                output_data[minute_str] = {
                    "prob": round(probs[idx] * 100, 1),
                    "runner_status": runner_status_str,
                }
            except KeyError:
                output_data[minute_str] = {
                    "prob": round(probs[idx] * 100, 1),
                    "runner_status": runner_status_str,
                }

        # --- 保存ディレクトリ作成 ---
        os.makedirs(f"data/LRA_data/LRA_{do_place}", exist_ok=True)

        # --- JSONファイル保存（予測結果） ---
        with open(
            f"data/LRA_data/LRA_{do_place}/{gamepk}_highlight_predictions_{do_place}_with_event_info.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

            # --- JSONファイル保存（回帰係数） ---
        if len(weights) > 0:
            with open(
                f"data/LRA_data/LRA_{do_place}/{gamepk}_weights_{do_place}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "feature_names": feature_names,
                        "feature_weights": weights.tolist(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        print(f"✅ 出力完了: {gamepk} - {do_place}")
