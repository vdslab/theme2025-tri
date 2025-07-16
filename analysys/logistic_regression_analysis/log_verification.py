import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

# --- 設定 ---
gamepk = "778199"
feature_group = "all_feature"
weights_path = f"data/LRA_data/LRA_{feature_group}/{gamepk}_weights_{feature_group}.json"
annotation_path = "data/anotation_data/cluster_3.csv"
molded_data_path = f"data/molded_data/{gamepk}_molded_data.json"

# --- データ読み込み ---
with open(weights_path, encoding="utf-8") as f:
    weights_json = json.load(f)

feature_names = weights_json["feature_names"]
print(f"🧩 特徴量数: {len(feature_names)}")
print("📌 使用特徴量:", feature_names)

# --- アノテーション読み込み ---
annotation_df = pd.read_csv(annotation_path, dtype={"gamePK": str})
annotation_df = annotation_df.set_index("gamePK").T
annotation_df.index = annotation_df.index.map(lambda x: f"{int(x):03d}")

if gamepk not in annotation_df.index:
    raise ValueError(f"{gamepk} のアノテーションが存在しません")

annotation_series = annotation_df.loc[gamepk].dropna()
annotation_dict = annotation_series.astype(int).to_dict()

# --- 試合データ読み込み ---
with open(molded_data_path, encoding="utf-8") as f:
    molded_data = json.load(f)

minutes_data = molded_data["minutes"]

# --- 特徴量構築 ---
X, y = [], []

for minute_str, minute_list in minutes_data.items():
    if not minute_list or minute_str not in annotation_dict:
        continue

    minute = minute_list[0]
    play = minute["play_features"]
    situ = minute["situation_features"]
    features = []

    # 再構成：all_feature の展開
    for key in ["single", "double", "triple", "home_run"]:
        features.append(int(play["hit_event"].get(key, False)))
    for key in ["regular_rbi", "tie_rbi", "go_ahead_rbi", "sayonara_rbi"]:
        features.append(int(play["rbi_impact"].get(key, False)))

    for key in [
        "none", "first", "second", "third",
        "first-second", "first-third", "second-third", "first-second-third"
    ]:
        features.append(int(situ["runner_status"].get(key, False)))

    for key in [
        "minus_less_3", "minus_2", "minus_1", "tie",
        "plus_1", "plus_2", "plus_more_3"
    ]:
        features.append(int(situ["score_difference"].get(key, False)))

    features.append(int(situ["inning_phase"].get("early", False)))

    X.append(features)
    y.append(annotation_dict[minute_str])

X = np.array(X)
y = np.array(y)

print(f"🎯 有効サンプル数: {len(X)}")

# --- モデル学習・評価 ---
model = LogisticRegression(max_iter=1000)

# ⬛️ 評価①: ホールドアウト検証（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- 📊 ホールドアウト評価（testデータ） ---")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ⬛️ 評価②: 全体学習 + 回帰係数の確認
model.fit(X, y)
print("\n--- ⚙️ 回帰係数と対応特徴量 ---")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"{name:30s}: {coef:.4f}")

# ⬛️ 評価③: 5分割 交差検証
cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring='f1')
print("\n--- 🔁 5分割交差検証 (F1スコア) ---")
print("各foldスコア:", np.round(cv_scores, 3))
print("平均F1スコア:", round(cv_scores.mean(), 3))
