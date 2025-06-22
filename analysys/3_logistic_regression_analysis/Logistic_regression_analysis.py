import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- 1. データの読み込み ---
with open("data/777490_processed_for_ra_data.json", encoding="utf-8") as f:
    data = json.load(f)


# --- 2. 1分単位特徴量フラット化 ---
def flatten_features(minute_data):
    features = []
    # hit_event
    for key in ["single", "double", "triple", "home_run"]:
        features.append(int(minute_data["play_features"]["hit_event"].get(key, False)))
    # rbi_impact
    for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
        features.append(int(minute_data["play_features"]["rbi_impact"].get(key, False)))
    # runner_status
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
        features.append(
            int(minute_data["situation_features"]["runner_status"].get(key, False))
        )
    # is_goahead_runner_on_base
    features.append(
        int(minute_data["situation_features"].get("is_goahead_runner_on_base", False))
    )
    # score_difference
    for key in [
        "minus_less_3",
        "minus_2",
        "minus_1",
        "tie",
        "plus_1",
        "plus_2",
        "plus_more_3",
    ]:
        features.append(
            int(minute_data["situation_features"]["score_difference"].get(key, False))
        )
    # inning_phase
    features.append(
        int(minute_data["situation_features"]["inning_phase"].get("early", False))
    )
    # diff_time
    features.append(float(minute_data["time"].get("diff_time", 0.0)))
    return features


# --- 3. 試合内すべての分の特徴量をDataFrame化 ---
def match_to_df(match_dict):
    rows = []
    for minute_key in sorted(match_dict.keys(), key=int):
        rows.append(flatten_features(match_dict[minute_key]))
    return pd.DataFrame(rows)


# --- 4. 集約特徴量 ---
def aggregate_features(df):
    agg = {}
    agg["sum"] = df.sum().values
    agg["mean"] = df.mean().values
    agg["max"] = df.max().values
    agg["var"] = df.var().values
    return np.concatenate([agg["sum"], agg["mean"], agg["max"], agg["var"]])


# --- 5. 1分単位で「ハイライトありかどうか」を判定 ---

# def is_minute_highlight(minute_data):
#     # play_features, situation_features に含まれるboolを全部調べて、1つでもTrueがあればTrue
#     # play_features の bool をすべて取得
#     for feature_group in ['hit_event', 'rbi_impact']:
#         for v in minute_data['play_features'][feature_group].values():
#             if v:
#                 return True
#     # situation_features の bool をすべて取得
#     for feature_group in ['runner_status', 'score_difference']:
#         for v in minute_data['situation_features'][feature_group].values():
#             if v:
#                 return True
#     # is_goahead_runner_on_base はbool単体
#     if minute_data['situation_features'].get('is_goahead_runner_on_base', False):
#         return True
#     # inning_phase の early はbool単体
#     if minute_data['situation_features']['inning_phase'].get('early', False):
#         return True
#     return False

# def is_minute_highlight(minute_data):
#     # play_features に含まれる特徴量のうち1つでも True であれば True
#     for feature_group in ['hit_event', 'rbi_impact']:
#         for v in minute_data['play_features'][feature_group].values():
#             if v:
#                 return True
#     return False


# def is_minute_highlight(minute_data):
#     # situation_features に含まれる特徴量のうち1つでも True であれば True
#     for feature_group in [
#         "runner_status",
#         "is_goahead_runner_on_base",
#         "score_difference",
#         "inning_phase",
#     ]:
#         for v in minute_data["situation_features"][feature_group].values():
#             if v:
#                 return True
#     return False


def is_minute_highlight(minute_data):
    # situation_features のうち、is_goahead_runner_on_base が True のものを返す
    return minute_data["situation_features"].get("is_goahead_runner_on_base", False)


# --- 6. データセット作成 ---
X = []
y = []

for match_id in data:
    df = match_to_df(data[match_id])
    agg_feat = aggregate_features(df)
    X.append(agg_feat)

    # 試合単位のラベル判定（1分でもTrueなら1）
    minutes = data[match_id]
    highlight_flag = any(is_minute_highlight(minutes[m]) for m in minutes)
    y.append(int(highlight_flag))

X = np.array(X)
X = np.nan_to_num(X, nan=0.0)  # NaN を 0 に置き換え
y = np.array(y)

print(f"Number of samples: {len(X)}")
print(f"Shape of each sample: {X[0].shape}")
print(f"Labels distribution: {np.bincount(y)}")

# --- 7. モデル学習・評価 ---
if len(np.unique(y)) > 1:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

    probs = model.predict_proba(X)[:, 1]
    print("Highlight probabilities:", probs)
else:
    print(
        "Warning: Only one class present in y. Logistic regression cannot be trained."
    )
