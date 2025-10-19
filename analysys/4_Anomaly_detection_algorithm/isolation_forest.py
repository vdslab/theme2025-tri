import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # IsolationForestから変更
from sklearn.metrics import classification_report, f1_score
# --- 対象試合リスト ---
pk_list = [
    778199,
    777579,
    777863,
    777940,
    777571,
    777988,
    778434,
    777701,
    778444,
    777649,
    778220,
    778406,
    778544,
    777726,
    778285,
    778262,
    778163,
    777505,
]


# --- ヘルパー関数 (変更なし) ---
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


def build_feature_df_with_context(data, group_paths, window=7):
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
    df_full = pd.concat(frames, axis=1).fillna(False).astype(int)
    return df_full


# --- メイン処理 ---
def main():
    # --- 1. データ読み込みと準備 ---
    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)

    all_features_list = []
    all_labels_list = []

    print("Loading and processing data for all games...")
    for gamepk in pk_list:
        gamepk_str = str(gamepk)
        molded_path = f"data/molded_data/{gamepk}_molded_data.json"
        try:
            with open(molded_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[SKIP] {gamepk}: molded_data not found.")
            continue

        all_plays = [play for minute in data["minutes"].values() for play in minute]
        if not all_plays:
            continue

        if gamepk_str not in annotation_df.columns:
            print(f"[SKIP] {gamepk}: Annotation column not found.")
            continue

        group_paths = extract_all_feature_paths(all_plays[0])
        features_df = build_feature_df_with_context(all_plays, group_paths, window=7)

        labels_series = annotation_df.set_index("gamePK")[gamepk_str]
        combined_df = features_df.copy()
        combined_df["label"] = labels_series
        combined_df.dropna(subset=["label"], inplace=True)  # ラベルがない行のみ削除

        if combined_df.empty:
            continue

        labels_final = combined_df.pop("label").astype(int)
        features_final = combined_df
        features_final["gamepk"] = gamepk

        all_features_list.append(features_final)
        all_labels_list.append(labels_final)

    if not all_features_list:
        print("No processable data found.")
        return

    # 全てのデータを結合
    X_all = pd.concat(all_features_list)
    y_all = pd.concat(all_labels_list)

    # --- 2. 学習データとテストデータの分割 ---
    print("\nSplitting data into training and testing sets...")
    split_point = int(len(pk_list) * 0.75)
    train_pks = set(pk_list[:split_point])
    test_pks = set(pk_list[split_point:])

    train_mask = X_all["gamepk"].isin(train_pks)
    test_mask = X_all["gamepk"].isin(test_pks)

    X_train = X_all[train_mask].drop(columns=["gamepk"])
    y_train = y_all[train_mask]
    X_test = X_all[test_mask].drop(columns=["gamepk"])
    y_test = y_all[test_mask]

    # NaNを最終処理
    X_train = X_train.fillna(False).astype(int)
    X_test = X_test.fillna(False).astype(int)

    if X_train.empty or X_test.empty:
        print("[SKIP] Train or Test set is empty after processing.")
        return

    print(f"Training set size: {len(X_train)} plays")
    print(f"Testing set size: {len(X_test)} plays")

    # # --- 3. contaminationの動的設定 ---
    # # 学習データ内の異常（興奮するプレイ=1）の割合を計算
    # actual_contamination = y_train.value_counts(normalize=True).get(
    #     1, 0.1
    # )  # ラベル1の割合
    # print(f"\nActual anomaly rate in training data: {actual_contamination:.2%}")
    # print("This will be used as the 'contamination' parameter.")

    # # --- 4. IsolationForestの学習と予測 ---
    # print("Training Isolation Forest model...")
    # model = IsolationForest(
    #     contamination=actual_contamination, n_estimators=100, random_state=42, n_jobs=-1
    # )

    # # 学習データのみでモデルを学習
    # model.fit(X_train)

    # # テストデータで予測
    # print("Predicting on the test set...")
    # # 予測結果は -1 (異常) or 1 (正常)
    # y_pred_iso = model.predict(X_test)

    # # 評価のためにラベル形式を統一 (異常:-1 -> 1, 正常:1 -> 0)
    # y_pred = np.where(y_pred_iso == -1, 1, 0)

    # # --- 5. 結果の評価 ---
    # print("\n" + "=" * 20 + " EVALUATION RESULTS " + "=" * 20)
    # print("'興奮するプレイ(1)' is treated as the 'anomaly' class.\n")

    # # classification_reportで全体像を表示
    # print(
    #     classification_report(
    #         y_test, y_pred, target_names=["普通のプレイ (0)", "興奮するプレイ (1)"]
    #     )
    # )
     # --- 3. RandomForestClassifierによる学習と予測 ---
    print("\nSolving as a Supervised Classification problem...")
    print("Training RandomForestClassifier model...")
    
    model = RandomForestClassifier(
        n_estimators=200,          # 木の数は多めが安定
        class_weight='balanced',   # 不均衡データに対応
        random_state=42,
        n_jobs=-1,
        max_depth=10               # 過学習防止のため深さを制限
    )
    
    # 学習データでモデルを学習
    model.fit(X_train, y_train)
    
    # テストデータで予測
    print("Predicting on the test set...")
    y_pred = model.predict(X_test)
    
    # --- 4. 結果の評価 ---
    print("\n" + "="*20 + " EVALUATION RESULTS (Classifier) " + "="*20)
    print(classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)']))

    # (試合ごとの評価も同様)

    # 試合ごとの精度も計算
    print("\n" + "-" * 20 + " Per-Game Performance " + "-" * 20)
    test_gamepks_series = X_all[test_mask]["gamepk"]
    for gamepk in sorted(list(test_pks)):
        game_mask = (test_gamepks_series == gamepk).values

        y_true_game = y_test[game_mask]
        y_pred_game = y_pred[game_mask]

        if len(y_true_game) == 0:
            continue

        f1 = f1_score(y_true_game, y_pred_game, zero_division=0)
        print(f"GamePK {gamepk}: F1-Score = {f1:.2%}")


if __name__ == "__main__":
    main()
