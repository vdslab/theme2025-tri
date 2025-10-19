import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# --- 対象試合リスト ---
pk_list = [
    778199, 777579, 777863, 777940, 777571, 777988,
    778062, 778434, 777701, 778444, 777649, 778220,
    778406, 778544, 777726, 778285, 778262, 778163, 777505
]

# --- ヘルパー関数 ---

def extract_all_feature_paths(play):
    """JSONから特徴量のパスを再帰的に抽出する"""
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
    """データと指定されたwindowサイズから特徴量DataFrameを構築する"""
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
def create_interaction_features(df):
    """
    【改訂版】提供されたJSON構造に最適化した特徴量生成関数
    """
    print("Creating interaction features based on actual data structure...")
    
    # --- 基本的な状況（コンテキスト）を定義 ---
    
    # 状況1: 試合の終盤か？
    # .get()を使い、'late'の列が存在しない場合は0(False)として扱う
    df['context.is_late_inning'] = df.get('cur.situation_features.inning_phase.late', 0)
    
    # 状況2: 接戦か？ (同点、1点リード、1点ビハインド)
    is_close_game = (
        df.get('cur.situation_features.score_difference.tie', 0) |
        df.get('cur.situation_features.score_difference.plus_1', 0) |
        df.get('cur.situation_features.score_difference.minus_1', 0)
    ).astype(int)
    df['context.is_close_game'] = is_close_game
    
    # 状況3: 得点圏にランナーがいるか？ (ランナー2塁 or 3塁)
    is_risp = (
        df.get('cur.situation_features.runner_status.second', 0) |
        df.get('cur.situation_features.runner_status.third', 0) |
        df.get('cur.situation_features.runner_status.first-second', 0) |
        df.get('cur.situation_features.runner_status.first-third', 0) |
        df.get('cur.situation_features.runner_status.second-third', 0) |
        df.get('cur.situation_features.runner_status.first-second-third', 0)
    ).astype(int)
    df['context.is_risp'] = is_risp

    # --- 状況を組み合わせた「興奮する場面」を定義 ---

    # 場面1: 終盤の接戦
    df['scenario.late_and_close'] = (df['context.is_late_inning'] & df['context.is_close_game']).astype(int)
    
    # 場面2: 満塁 (キーを 'first-second-third' に修正)
    df['scenario.bases_loaded'] = df.get('cur.situation_features.runner_status.first-second-third', 0)

    # --- 「場面」と「プレイ結果」を組み合わせた最終的な特徴量を作成 ---

    # 特徴量1: 終盤の接戦で勝ち越し打
    is_go_ahead_rbi = df.get('cur.play_features.rbi_impact.go_ahead_rbi', 0)
    df['interaction.clutch_go_ahead_rbi'] = (df['scenario.late_and_close'] & is_go_ahead_rbi).astype(int)
    
    # 特徴量2: 終盤の接戦で同点打
    is_tie_rbi = df.get('cur.play_features.rbi_impact.tie_rbi', 0)
    df['interaction.clutch_tie_rbi'] = (df['scenario.late_and_close'] & is_tie_rbi).astype(int)

    # 特徴量3: 得点圏での長打 (二塁打 or 三塁打 or 本塁打)
    is_long_hit = (
        df.get('cur.play_features.hit_event.double', 0) |
        df.get('cur.play_features.hit_event.triple', 0) |
        df.get('cur.play_features.hit_event.home_run', 0)
    ).astype(int)
    df['interaction.risp_long_hit'] = (df['context.is_risp'] & is_long_hit).astype(int)
    
    # 特徴量4: 満塁でのホームラン (最高に興奮するプレイの一つ)
    is_home_run = df.get('cur.play_features.hit_event.home_run', 0)
    df['interaction.grand_slam'] = (df['scenario.bases_loaded'] & is_home_run).astype(int)

    # 特徴量5: サヨナラ打 (これ自体が最高の興奮シーン)
    # .get()で安全にアクセスする
    df['interaction.sayonara'] = df.get('cur.play_features.rbi_impact.sayonara_rbi', 0)

    print(f"Created new features. Total features now: {len(df.columns)}")
    return df

# --- メイン処理 ---
def main():
    window = 1
    print(f"\n{'='*25} RUNNING WITH FIXED WINDOW SIZE: {window} {'='*25}")
    
    # --- 1. データ準備フェーズ ---
    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)

    all_features_list = []
    all_labels_list = []
    for gamepk in pk_list:
        gamepk_str = str(gamepk)
        molded_path = f"data/molded_data/{gamepk_str}_molded_data.json"
        try:
            with open(molded_path, encoding="utf-8") as f: data = json.load(f)
        except FileNotFoundError: continue
        
        all_plays = [play for minute in data["minutes"].values() for play in minute]
        if not all_plays: continue
        if gamepk_str not in annotation_df.columns: continue
        
        group_paths = extract_all_feature_paths(all_plays[0])
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
        print("[SKIP] No processable data found.")
        return

    X_all = pd.concat(all_features_list)
    y_all = pd.concat(all_labels_list)
    split_point = int(len(pk_list) * 0.75)
    train_pks = set(pk_list[:split_point])
    test_pks = set(pk_list[split_point:])
    train_mask = X_all['gamepk'].isin(train_pks)
    test_mask = X_all['gamepk'].isin(test_pks)
    X_train_orig = X_all[train_mask].drop(columns=['gamepk'])
    y_train = y_all[train_mask]
    X_test_orig = X_all[test_mask].drop(columns=['gamepk'])
    y_test = y_all[test_mask]
    
    if X_train_orig.empty or X_test_orig.empty:
        print("[SKIP] Train or Test set is empty.")
        return
    
    # --- 2. 特徴量エンジニアリング ---
    X_train = create_interaction_features(X_train_orig.copy())
    X_test = create_interaction_features(X_test_orig.copy())
    
    # --- 3. グリッドサーチによるパラメータチューニング ---
    print("\nStarting Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("\nBest parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # --- 4. 近接性アプローチ (最適化されたモデルを使用) ---
    print("\nStarting anomaly detection based on proximity using the best model...")
    leaf_indices = best_model.apply(X_test)
    n_samples = X_test.shape[0]

    print(f"Calculating {n_samples}x{n_samples} proximity matrix...")
    proximity_matrix = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples), desc="Proximity Calculation"):
        for j in range(i, n_samples):
            proximity = np.sum(leaf_indices[i, :] == leaf_indices[j, :])
            proximity_matrix[i, j] = proximity
            proximity_matrix[j, i] = proximity
    proximity_matrix /= best_model.n_estimators

    avg_proximity = np.sum(proximity_matrix, axis=1) / n_samples
    raw_scores = 1 / (avg_proximity + 1e-9)

    contamination_rate = y_test.value_counts(normalize=True).get(1, 0.1)
    threshold = np.percentile(raw_scores, 100 * (1 - contamination_rate))
    y_pred = (raw_scores >= threshold).astype(int)
    
    # --- 5. 結果表示 ---
    print(f"\n--- Overall Results (Features + Tuning + Proximity) ---")
    print(classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)'], zero_division=0))

    # --- 6. 試合ごとの結果表示 & 詳細保存 ---
    print("--- Per-Game Results ---")
    os.makedirs("results", exist_ok=True)
    target_gamepk = 778163
    test_gamepks = X_all[test_mask]['gamepk']

    for gamepk in sorted(list(test_pks)):
        game_mask = (test_gamepks == gamepk).values
        y_game = y_test[game_mask]
        if y_game.empty: continue
            
        y_pred_game = y_pred[game_mask]
        
        accuracy = accuracy_score(y_game, y_pred_game)
        precision = precision_score(y_game, y_pred_game, zero_division=0)
        recall = recall_score(y_game, y_pred_game, zero_division=0)
        f1 = f1_score(y_game, y_pred_game, zero_division=0)
        print(f"[{gamepk}] Acc={accuracy:.2%}, Prec={precision:.2%}, Rec={recall:.2%}, F1={f1:.2%}")

        if gamepk == target_gamepk:
            result_df = pd.DataFrame({
                "minute": range(1, len(y_game) + 1),
                "y_true": y_game.values,
                "y_pred": y_pred_game,
                "anomaly_score": raw_scores[game_mask]
            })
            result_path = f"results/game_{gamepk}_window{window}_tuned_proximity_predictions.csv"
            result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
            print(f"Saved detailed predictions for gamepk {gamepk} -> {result_path}")
    
    # --- 7. 特徴量の重要度 ---
    importances_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n▼ 予測に重要だった特徴量 Top 20")
    print(importances_df.head(20))


if __name__ == "__main__":
    main()