import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

# 複数 gamePK を使用する
pk_list = [
    "778199", "777579", "777863", "777940", "777571", "777988",
    "778062", "778434", "777701", "778444", "777649", "778220",
    "778406", "778544", "777726", "778285", "778262", "778163", "777505",
]

feature_groups = [
    "all_feature", "play_features", "situation_features",
    "hit_event", "rbi_impact", "runner_status", "score_difference", "inning_phase"
]

annotation_df = pd.read_csv("data/anotation_data/cluster_3.csv", dtype={"gamePK": str})
annotation_df = annotation_df.set_index("gamePK").T
annotation_df.index = annotation_df.index.map(lambda x: f"{int(x):03d}")
print(annotation_df.index)

for do_place in feature_groups:
    print(f"\n===== 特徴群: {do_place} =====")
    X_all, y_all, feature_names = [], [], None

    for gamepk in pk_list:
        if gamepk not in annotation_df.index:
            print(f"⚠️ アノテーションが存在しない: {gamepk}")
            continue

        annotation_series = annotation_df.loc[gamepk].dropna()
        annotation_dict = annotation_series.astype(int).to_dict()

        try:
            with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
                molded_data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 試合データが見つかりません: {gamepk}")
            continue

        minutes_data = molded_data["minutes"]

        def flatten_features(minute):
            play = minute["play_features"]
            situ = minute["situation_features"]
            features = []
            names = []

            def add_feature(val, name):
                features.append(int(val))
                names.append(name)

            if do_place in ["hit_event", "play_features", "all_feature"]:
                for key in ["single", "double", "triple", "home_run"]:
                    add_feature(play["hit_event"].get(key, False), f"hit_event_{key}")

            if do_place in ["rbi_impact", "play_features", "all_feature"]:
                for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
                    add_feature(play["rbi_impact"].get(key, False), f"rbi_impact_{key}")

            if do_place in ["runner_status", "situation_features", "all_feature"]:
                for key in ["none", "first", "second", "third", "first-second", "first-third", "second-third", "first-second-third"]:
                    add_feature(situ["runner_status"].get(key, False), f"runner_status_{key}")

            if do_place in ["score_difference", "situation_features", "all_feature"]:
                for key in ["minus_less_3", "minus_2", "minus_1", "tie", "plus_1", "plus_2", "plus_more_3"]:
                    add_feature(situ["score_difference"].get(key, False), f"score_difference_{key}")

            if do_place in ["inning_phase", "situation_features", "all_feature"]:
                add_feature(situ["inning_phase"].get("early", False), "inning_phase_early")

            return features, names

        for minute_str, minute_list in minutes_data.items():
            if not minute_list or minute_str not in annotation_dict:
                continue
            minute = minute_list[0]
            features, names = flatten_features(minute)
            X_all.append(features)
            y_all.append(annotation_dict[minute_str])
            if feature_names is None:
                feature_names = names

    if not X_all:
        print(f"⚠️ 有効なデータが存在しない: {do_place}")
        continue

    X_all = np.nan_to_num(np.array(X_all))
    y_all = np.array(y_all)

    if len(np.unique(y_all)) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_all, y_all)
        weights = model.coef_[0]
        weight_dict = {name: round(w, 4) for name, w in zip(feature_names, weights)}
        sorted_weights = dict(sorted(weight_dict.items(), key=lambda x: abs(x[1]), reverse=True))

        os.makedirs(f"data/LRA_data/LRA_{do_place}", exist_ok=True)
        with open(f"data/LRA_data/LRA_{do_place}/global_weights_{do_place}.json", "w", encoding="utf-8") as f:
            json.dump(sorted_weights, f, ensure_ascii=False, indent=2)
        print(f"✅ 学習・保存完了: {do_place}")
    else:
        print(f"⚠️ ラベルが1種類しかないためモデル学習不可: {do_place}")
