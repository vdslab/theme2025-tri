import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

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

def build_feature_df_with_context(data, group_paths, window=1): # windowをデフォルト1に
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

# --- メイン処理 ---
def main():
    # ★ 変更点: windowサイズを1に固定
    window = 1
    print(f"\n{'='*25} RUNNING WITH FIXED WINDOW SIZE: {window} {'='*25}")
    
    # アノテーション(csv)データ読み込み
    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)

    all_features_list = []
    all_labels_list = []

    # --- 1. データ準備フェーズ ---
    for gamepk in pk_list:
        gamepk_str = str(gamepk)
        molded_path = f"data/molded_data/{gamepk_str}_molded_data.json"
        try:
            with open(molded_path, encoding="utf-8") as f: data = json.load(f)
        except FileNotFoundError:
            print(f"[{gamepk_str}] [SKIP] molded_data not found.")
            continue
        
        all_plays = [play for minute in data["minutes"].values() for play in minute]
        if not all_plays: continue

        if gamepk_str not in annotation_df.columns:
            print(f"[{gamepk_str}] [SKIP] Annotation column not found.")
            continue
        
        group_paths = extract_all_feature_paths(all_plays[0])
        # window=1 を渡す
        features_df = build_feature_df_with_context(all_plays, group_paths, window=window)
        
        labels_series = annotation_df.set_index("gamePK")[gamepk_str]
        combined_df = features_df.copy()
        combined_df['label'] = labels_series
        combined_df.dropna(inplace=True)
        
        if combined_df.empty: continue
            
        labels_final = combined_df.pop('label').astype(int)
        features_final = combined_df
        features_final['gamepk'] = gamepk
        all_features_list.append(features_final)
        all_labels_list.append(labels_final)

    if not all_features_list:
        print(f"[SKIP] No processable data found.")
        return

    X_all = pd.concat(all_features_list)
    y_all = pd.concat(all_labels_list)

    split_point = int(len(pk_list) * 0.75)
    train_pks = set(pk_list[:split_point])
    test_pks = set(pk_list[split_point:])

    train_mask = X_all['gamepk'].isin(train_pks)
    test_mask = X_all['gamepk'].isin(test_pks)

    X_train_orig = X_all[train_mask].drop(columns=['gamepk'])
    y_train_orig = y_all[train_mask]
    X_test = X_all[test_mask].drop(columns=['gamepk'])
    y_test = y_all[test_mask]
    
    if X_train_orig.empty or X_test.empty:
        print(f"[SKIP] Train or Test set is empty.")
        return
        
    # --- 2. ★★★ One-Class分類のための学習データ再構築 ★★★ ---
    print("\nRebuilding training data for One-Class classification...")

    # 2-1. 訓練データから「正常なプレイ（ラベル0）」のみを抽出
    X_train_normal = X_train_orig[y_train_orig == 0]
    y_train_normal = y_train_orig[y_train_orig == 0]
    print(f"Found {len(X_train_normal)} 'normal' plays for training.")

    # 2-2. 「人工的な異常データ」を生成
    # 正常データと同じ数のランダムなデータを生成する
    n_synthetic_samples = len(X_train_normal)
    n_features = X_train_normal.shape[1]
    # 元の特徴量が0か1なので、それに合わせてランダムな0/1データを生成
    X_synthetic_outliers = pd.DataFrame(
        np.random.randint(0, 2, size=(n_synthetic_samples, n_features)),
        columns=X_train_normal.columns
    )
    y_synthetic_outliers = pd.Series(np.ones(n_synthetic_samples, dtype=int)) # ラベルは1
    print(f"Generated {len(X_synthetic_outliers)} synthetic 'outlier' plays.")

    # 2-3. 「正常データ」と「人工的な異常データ」を結合して新しい訓練データを作成
    X_train_one_class = pd.concat([X_train_normal, X_synthetic_outliers], ignore_index=True)
    y_train_one_class = pd.concat([y_train_normal, y_synthetic_outliers], ignore_index=True)
    print(f"New one-class training set size: {len(X_train_one_class)} samples.")
    # --- ここまでがOne-Class分類のための変更点 ---

    # --- 3. 学習・評価フェーズ ---
    print("\nTraining RandomForest model...")
    # class_weightは不均衡データ用なので、今回は均等にしたため不要だが念のため残す
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # ★変更点：再構築したデータで学習
    model.fit(X_train_one_class, y_train_one_class)
    
    # 評価は元のテストデータ(X_test, y_test)で行う
    y_pred = model.predict(X_test)
    
    # --- 4. 結果表示 ---
    print(f"\n--- Overall Results for window = {window} (One-Class Method) ---")
    print(classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)'], zero_division=0))

    # --- 試合ごとの結果表示 ---
    print("--- Per-Game Results ---")
    
    # 結果保存用のディレクトリを作成
    os.makedirs("results", exist_ok=True)
    target_gamepk = 778163  # 保存したい試合番号

    for gamepk in sorted(list(test_pks)): # ソートして順序を固定
        # gamepkでフィルタリングするためのマスクを X_all から再作成
        game_mask_in_all = X_all['gamepk'] == gamepk
        # test_mask とのANDを取ることで、確実にテストセット内の試合のみを対象にする
        game_mask_in_test = game_mask_in_all & test_mask

        X_game = X_all[game_mask_in_test].drop(columns=['gamepk'])
        y_game = y_all[game_mask_in_test]

        if X_game.empty:
            continue

        y_pred_game = model.predict(X_game)
        y_prob_game = model.predict_proba(X_game)[:, 1]

        accuracy = accuracy_score(y_game, y_pred_game)
        precision = precision_score(y_game, y_pred_game, zero_division=0)
        recall = recall_score(y_game, y_pred_game, zero_division=0)
        f1 = f1_score(y_game, y_pred_game, zero_division=0)

        print(f"[{gamepk}] Acc={accuracy:.2%}, Prec={precision:.2%}, Rec={recall:.2%}, F1={f1:.2%}")

        # 保存対象のgamepkならCSV出力
        if gamepk == target_gamepk:
            # minuteのインデックスをリセットして1から始める
            y_game_reset = y_game.reset_index(drop=True)
            
            result_df = pd.DataFrame({
                "minute": range(1, len(y_game_reset) + 1),
                "y_true": y_game_reset.values,
                "y_pred": y_pred_game,
                "prob_exciting": y_prob_game
            })
            result_path = f"results/game_{gamepk}_window{window}_one_class_predictions.csv"
            result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
            print(f"Saved detailed predictions for gamepk {gamepk} -> {result_path}")
    
    # --- 5. 特徴量の重要度 ---
    importances = pd.Series(model.feature_importances_, index=X_train_one_class.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    print("\n▼ 予測に重要だった特徴量 Top 20")
    print(top20)

if __name__ == "__main__":
    main()