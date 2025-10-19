import json
import numpy as np
import pandas as pd
import rrcf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

# --- 対象試合リスト (変更なし) ---
pk_list = [
    778199, 777579, 777863, 777940, 777571, 777988,
    778062, 778434, 777701, 778444, 777649, 778220,
    778406, 778544, 777726, 778285, 778262, 778163, 777505
]


def extract_all_feature_paths(play):
    """
    (変更なし)
    ネストされた辞書から特徴量のパスを再帰的に抽出する関数
    """
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
    """
    (変更なし)
    データと特徴量パスリストから、時間的文脈(window)を考慮した特徴量DataFrameを構築する関数
    """
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
    # --- 1. データ準備フェーズ (変更なし) ---
    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)
    window_sizes_to_test = range(1, 11)
    final_summary = []
    if not os.path.exists("results"):
        os.makedirs("results")

    for window in window_sizes_to_test:
        print(f"\n{'='*25} TESTING WINDOW SIZE: {window} {'='*25}")
        
        all_features_list = []
        all_labels_list = []

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
            features_df = build_feature_df_with_context(all_plays, group_paths, window=window)
            
            labels_series = annotation_df[gamepk_str]
            labels_series.index = labels_series.index + 1
            
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
            print(f"[SKIP] No processable data for window size {window}.")
            continue

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
            print(f"[SKIP] Train or Test set is empty for window size {window}.")
            continue
        
        # --- Random Cut Forest の実装 ---
        num_trees = 100
        tree_size = 256
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)
            
        train_scores = []
        X_train_np = X_train.values
        for index, point in enumerate(X_train_np):
            for tree in forest:
                if len(tree.leaves) > tree_size:
                    oldest_index = min(tree.leaves.keys())
                    tree.forget_point(oldest_index)
                tree.insert_point(point, index=index)
            
            avg_codisp = 0
            for tree in forest:
                avg_codisp += tree.codisp(index)
            train_scores.append(avg_codisp / num_trees)
        
        threshold = np.percentile(train_scores, 99)
        print(f"Anomaly threshold set to: {threshold:.4f}")

        test_scores = []
        X_test_np = X_test.values
        start_index = len(X_train_np)
        for index_offset, point in enumerate(X_test_np):
            index = start_index + index_offset
            for tree in forest:
                if len(tree.leaves) > tree_size:
                    oldest_index = min(tree.leaves.keys())
                    tree.forget_point(oldest_index)
                tree.insert_point(point, index=index)
            
            avg_codisp = 0
            for tree in forest:
                avg_codisp += tree.codisp(index)
            test_scores.append(avg_codisp / num_trees)
        
        test_scores_np = np.array(test_scores)
        y_pred = (test_scores_np > threshold).astype(int)

        target_gamepk = 778163
        for gamepk in test_pks:
            game_mask = X_all['gamepk'] == gamepk
            final_mask = game_mask & test_mask
            if not final_mask.any(): continue
            
            X_game = X_all[final_mask].drop(columns=['gamepk'])
            y_game = y_all[final_mask]

            if X_game.empty: continue

            # ★修正点: IndexErrorを回避するため、より安全なブールマスク方式に変更
            # X_testのインデックスが現在の試合(X_game)のインデックスに含まれるかどうかのブールマスクを作成
            game_mask_in_test = X_test.index.isin(X_game.index)

            # ブールマスクを使って、テスト全体のスコアからこの試合に対応するスコアだけを抽出
            game_scores_np = test_scores_np[game_mask_in_test]
            
            # 念のため、抽出したスコアとラベルの数が一致するか確認
            if len(game_scores_np) != len(y_game):
                print(f"[{gamepk}] [SKIP] Mismatch between scores ({len(game_scores_np)}) and labels ({len(y_game)}) length.")
                continue
            
            y_pred_game = (game_scores_np > threshold).astype(int)
            
            accuracy = accuracy_score(y_game, y_pred_game)
            precision = precision_score(y_game, y_pred_game, zero_division=0)
            recall = recall_score(y_game, y_pred_game, zero_division=0)
            f1 = f1_score(y_game, y_pred_game, zero_division=0)

            print(f"[{gamepk}] Acc={accuracy:.2%}, Prec={precision:.2%}, Rec={recall:.2%}, F1={f1:.2%}")

            if gamepk == target_gamepk:
                result_df = pd.DataFrame({
                    "minute": y_game.index,
                    "y_true": y_game.values,
                    "y_pred": y_pred_game,
                    "anomaly_score": game_scores_np
                })
                result_path = f"results/game_{gamepk}_window_{window}_predictions_rcf.csv"
                result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
                print(f"Saved detailed predictions for gamepk {gamepk} -> {result_path}")
        
        # --- 3. 結果の記録 ---
        print(f"\n--- Results for window = {window} ---")
        print(classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)'], zero_division=0))
        report_dict = classification_report(y_test, y_pred, target_names=['普通のプレイ (0)', '興奮するプレイ (1)'], output_dict=True, zero_division=0)
        
        exciting_play_metrics = report_dict.get('興奮するプレイ (1)', {})
        summary_data = {
            'window_size': window,
            'f1_score': exciting_play_metrics.get('f1-score'),
            'recall': exciting_play_metrics.get('recall'),
            'precision': exciting_play_metrics.get('precision'),
            'accuracy': report_dict.get('accuracy')
        }
        final_summary.append(summary_data)
        
    # --- 4. 最終サマリーの表示 ---
    print("\n\n" + "="*25 + " FINAL SUMMARY (by Window Size) " + "="*25)
    if not final_summary:
        print("No results to summarize.")
    else:
        summary_df = pd.DataFrame(final_summary).set_index('window_size')
        print(summary_df.sort_values(by='f1_score', ascending=False))

if __name__ == "__main__":
    main()

