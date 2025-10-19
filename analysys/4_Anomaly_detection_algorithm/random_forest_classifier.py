import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss

# --- 対象試合リスト (変更なし) ---
pk_list = [
    778199, 777579, 777863, 777940, 777571, 777988,
    778062, 778434, 777701, 778444, 777649, 778220,
    778406, 778544, 777726, 778285, 778262, 778163, 777505
]


def extract_all_feature_paths(play):
    paths = []
    def recurse(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                if all(isinstance(val, bool) for val in v.values()):
                    paths.append(path + [k])
                recurse(v, path + [k])
    recurse(play, [])
    return paths


def build_feature_df_with_context(data, group_paths, window=1):
    dfs = []
    for play in data:
        combined = {}
        for path in group_paths:
            d = play
            for key in path:
                d = d.get(key, {})
            if isinstance(d, dict):
                for k, v in d.items():
                    combined[".".join(path + [k])] = v
        dfs.append(combined)

    df_main = pd.DataFrame(dfs).fillna(False).astype(int)
    frames = []
    for offset in range(-window, window + 1):
        df_shifted = df_main.shift(offset)
        label = "cur" if offset == 0 else ("prev" if offset < 0 else "next")
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    df_full = pd.concat(frames, axis=1)
    return df_full


def main():
    window = 1  # 固定
    print(f"=== Running model with window={window} ===")

    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)

    all_features_list = []
    all_labels_list = []

    for gamepk in pk_list:
        gamepk_str = str(gamepk)
        molded_path = f"data/molded_data/{gamepk_str}_molded_data.json"

        try:
            with open(molded_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[{gamepk_str}] [SKIP] molded_data not found.")
            continue

        all_plays = [play for minute in data["minutes"].values() for play in minute]
        if not all_plays:
            continue

        if gamepk_str not in annotation_df.columns:
            print(f"[{gamepk_str}] [SKIP] Annotation column not found.")
            continue

        group_paths = extract_all_feature_paths(all_plays[0])
        features_df = build_feature_df_with_context(all_plays, group_paths, window=window)
        labels_series = annotation_df.set_index("gamePK")[gamepk_str]

        combined_df = features_df.copy()
        combined_df['label'] = labels_series
        combined_df.dropna(inplace=True)
        if combined_df.empty:
            continue

        labels_final = combined_df.pop('label').astype(int)
        features_final = combined_df
        features_final['gamepk'] = gamepk

        all_features_list.append(features_final)
        all_labels_list.append(labels_final)

    if not all_features_list:
        print("[STOP] No processable data found.")
        return

    X_all = pd.concat(all_features_list)
    y_all = pd.concat(all_labels_list)

    split_point = int(len(pk_list) * 0.75)
    train_pks = set(pk_list[:split_point])
    test_pks = set(pk_list[split_point:])

    train_mask = X_all['gamepk'].isin(train_pks)
    test_mask = X_all['gamepk'].isin(test_pks)

    X_train = X_all[train_mask].drop(columns=['gamepk'])
    y_train = y_all[train_mask]
    X_test = X_all[test_mask].drop(columns=['gamepk'])
    y_test = y_all[test_mask]

    if X_train.empty or X_test.empty:
        print("[STOP] Train or Test set is empty.")
        return
    # パラメータ関連
    # n_estimators：森を構成する木の本数
    # max_depth：各決定木の最大深さ
    # min_samples_split：ノードを分割するために必要な最小サンプル数
    # min_samples_leaf：葉ノードに必要な最小サンプル数
    # max_features：各分割で考慮する特徴量の最大数
    # class_weight：クラスの重み付け方法
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n--- Overall Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)'], zero_division=0))

    # ========== AIC / BIC 評価 ==========
    n = len(y_test)
    k = X_train.shape[1]  # 特徴量数をパラメータ数の近似として使用
    log_likelihood = -log_loss(y_test, y_prob, normalize=False)  # 負の対数尤度

    AIC = 2 * k - 2 * log_likelihood
    BIC = k * np.log(n) - 2 * log_likelihood

    print("\n--- Information Criteria ---")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")

    # ===== Optional: 特定試合の予測保存 =====
    target_gamepk = 778163
    if target_gamepk in test_pks:
        mask = X_all['gamepk'] == target_gamepk
        X_game = X_all[mask].drop(columns=['gamepk'])
        y_game = y_all[mask]

        if not X_game.empty:
            y_pred_game = model.predict(X_game)
            y_prob_game = model.predict_proba(X_game)[:, 1]

            result_df = pd.DataFrame({
                "minute": range(1, len(y_game) + 1),
                "y_true": y_game.values,
                "y_pred": y_pred_game,
                "prob_exciting": y_prob_game
            })
            result_path = f"results/game_{target_gamepk}_{window}_predictions.csv"
            result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
            print(f"Saved predictions for gamepk {target_gamepk} -> {result_path}")

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    print("\n▼ 予測に重要だった特徴量 Top 20")
    print(top20)


if __name__ == "__main__":
    main()

