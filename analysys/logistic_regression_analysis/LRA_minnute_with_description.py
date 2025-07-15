import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pk_list = ["777398","777490","777824","777838","777854","777866"]  # 対象のゲームPK
# pk_list = ["777398"] テスト用

# パラメータ
# pk_list = ["777398", "777490", "777824", "777838", "777854", "777866"]
pk_list = ["777398"]  # テスト用
feature_groups = [
    "play_features", "situation_features", "hit_event", "rbi_impact",
    "runner_status", "score_difference", "inning_phase"
]
feature_elements = {
    "hit_event": ["single", "double", "triple", "home_run"],
    "rbi_impact": ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"],
    "runner_status": [
        "none", "first", "second", "third",
        "first-second", "first-third", "second-third", "first-second-third"
    ],
    "score_difference": [
        "minus_less_3", "minus_2", "minus_1", "tie",
        "plus_1", "plus_2", "plus_more_3"
    ],
    "inning_phase": ["early"],
}

# 重み候補
weight_values = [0.1, 1.0, 2.0, 5.0, 10.0]

# 特徴量抽出関数
def extract_features(minute, do_place, weight_map):
    play = minute["play_features"]
    situ = minute["situation_features"]
    features = []

    if do_place == "play_features":
        for sub in ["hit_event", "rbi_impact"]:
            features += [weight_map.get(k, 1.0) * int(play[sub].get(k, 0)) for k in feature_elements[sub]]
    elif do_place == "situation_features":
        for sub in ["runner_status", "score_difference", "inning_phase"]:
            source = situ[sub]
            features += [weight_map.get(k, 1.0) * int(source.get(k, 0)) for k in feature_elements[sub]]
    else:
        source = play[do_place] if do_place in play else situ[do_place]
        features += [weight_map.get(k, 1.0) * int(source.get(k, 0)) for k in feature_elements[do_place]]
    return features

# ハイライトラベル関数
def is_highlight(minute, do_place):
    play = minute["play_features"]
    situ = minute["situation_features"]
    if do_place == "play_features":
        return int(any(v for sub in ["hit_event", "rbi_impact"] for v in play[sub].values()))
    elif do_place == "situation_features":
        return int(any(v for sub in ["runner_status", "score_difference", "inning_phase"] for v in situ[sub].values()))
    elif do_place in play:
        return int(any(play[do_place].values()))
    elif do_place in situ:
        return int(any(situ[do_place].values()))
    return 0

# 実験実行
def run_experiment(do_place, molded_data, output_csv_path, output_plot_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if do_place not in feature_elements and do_place not in ["play_features", "situation_features"]:
        return  # 無効

    minutes_data = molded_data["minutes"]
    keys = []

    if do_place in ["play_features"]:
        keys = feature_elements["hit_event"] + feature_elements["rbi_impact"]
    elif do_place in ["situation_features"]:
        keys = feature_elements["runner_status"] + feature_elements["score_difference"] + feature_elements["inning_phase"]
    else:
        keys = feature_elements[do_place]

    all_combinations = product(weight_values, repeat=len(keys))
    results = []

    for comb in all_combinations:
        weight_map = dict(zip(keys, comb))
        X, y = [], []

        for minute_list in minutes_data.values():
            if not minute_list:
                continue
            minute = minute_list[0]
            X.append(extract_features(minute, do_place, weight_map))
            y.append(is_highlight(minute, do_place))

        X = np.nan_to_num(np.array(X))
        y = np.array(y)

        if len(np.unique(y)) <= 1:
            results.append((comb, 0))
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        high_conf = np.sum(probs > 0.9)
        results.append((comb, high_conf))

    # CSV出力
    with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys + ["high_conf_count"])
        for comb, count in results:
            writer.writerow(list(comb) + [count])
    print(f"✅ CSV保存完了: {output_csv_path}")

    # 最大の1パターンだけグラフ表示（次元が高すぎるため）
    best = max(results, key=lambda x: x[1])
    plt.figure(figsize=(10, 4))
    plt.bar(keys, best[0])
    plt.title(f"Best Weights for {do_place} (High prob > 0.9: {best[1]})")
    plt.ylabel("Weight")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"📊 グラフ保存完了: {output_plot_path}")

# メインループ
for gamepk in pk_list:
    with open(f"data/molded_data/{gamepk}_molded_data.json", encoding="utf-8") as f:
        molded_data = json.load(f)

    for do_place in feature_groups:
        if do_place == "situation_features":
            print(f"⏭️ スキップ: {do_place}")
            continue
        print(f"🎯 実験開始: {gamepk} - {do_place}")
        run_experiment(
            do_place,
            molded_data,
            output_csv_path=f"results/{gamepk}_{do_place}_results.csv",
            output_plot_path=f"results/{gamepk}_{do_place}_best_plot.png"
        )
