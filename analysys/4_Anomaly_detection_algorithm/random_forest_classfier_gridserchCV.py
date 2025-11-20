import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# classification_report の 'output_dict=True' を使うため、import文を少し変更
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
)  # GridSearchCV と GroupKFold をインポート

# --- 対象試合リスト (変更なし) ---
pk_list = [
    778199,
    777579,
    777863,
    777940,
    777571,
    777988,
    778062,
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


def extract_all_feature_paths(play):
    # (この関数は変更なし)
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
    # (この関数は変更なし)
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
    for offset in range(
        -window, 1
    ):  # window サイズに基づいて範囲を調整、現在は過去のみ参照
        df_shifted = df_main.shift(offset)
        label = "cur" if offset == 0 else ("prev" if offset < 0 else "next")
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    df_full = pd.concat(frames, axis=1)
    return df_full


def main():

    #   変更点 1: 試したい window サイズのリストを定義
    # === 変更箇所 1: 対象windowを1〜3に限定 ===
    window_sizes_to_test = [1, 2, 3]

    # 特徴量上位Nを指定 ===
    TOP_N_FEATURES = 50

    #   変更点 2: 最終結果を格納するためのリストを初期化
    results_summary = []

    #   変更点 3: window サイズごとにループ処理を行う
    for window in window_sizes_to_test:

        print(f"\n=======================================================")
        print(f"=== Running model with window={window} ===")
        print(f"=======================================================")

        annotation_path = "data/anotation_data/cluster_3.csv"
        annotation_df = pd.read_csv(annotation_path)

        all_features_list = []
        all_labels_list = []

        for gamepk in pk_list:
            gamepk_str = str(gamepk)
            molded_path = f"data/test_molded_data/{gamepk_str}_test2_molded_data.json"

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

            features_df = build_feature_df_with_context(
                all_plays, group_paths, window=window
            )
            labels_series = annotation_df.set_index("gamePK")[gamepk_str]

            combined_df = features_df.copy()
            combined_df["label"] = labels_series
            combined_df.dropna(inplace=True)
            if combined_df.empty:
                continue

            labels_final = combined_df.pop("label").astype(int)
            features_final = combined_df
            features_final["gamepk"] = gamepk

            all_features_list.append(features_final)
            all_labels_list.append(labels_final)

        if not all_features_list:
            print(f"[STOP] No processable data found for window={window}.")
            continue

        X_all = pd.concat(all_features_list)
        y_all = pd.concat(all_labels_list)

        split_point = int(len(pk_list) * 0.75)
        train_pks = set(pk_list[:split_point])
        test_pks = set(pk_list[split_point:])

        train_mask = X_all["gamepk"].isin(train_pks)
        test_mask = X_all["gamepk"].isin(test_pks)

        # === GridSearchCV 導入箇所 ===

        groups = X_all[train_mask]["gamepk"]
        X_train = X_all[train_mask].drop(columns=["gamepk"])
        y_train = y_all[train_mask]

        X_test = X_all[test_mask].drop(columns=["gamepk"])
        y_test = y_all[test_mask]

        if X_train.empty or X_test.empty:
            print(f"[STOP] Train or Test set is empty for window={window}.")
            continue

        param_grid = {
            "n_estimators": [200, 500, 900],  # 森を構成する木の本数
            "max_depth": [10, 15, 20],  # 各決定木の最大深さ
            "min_samples_split": [
                2,
                5,
                10,
            ],  # ノードを分割するために必要な最小サンプル数
            "min_samples_leaf": [1, 3, 5],  # 葉ノードに必要な最小サンプル数
            "max_features": [
                "sqrt",
                "log2",
                0.5,
                0.7,
                0.9,
            ],  # 各分割で考慮する特徴量の最大数
        }

        base_model = RandomForestClassifier(
            random_state=42, class_weight="balanced_subsample", n_jobs=-1
        )

        gkf = GroupKFold(n_splits=5)

        grid_search = GridSearchCV(
            estimator=base_model, param_grid=param_grid, scoring="f1", cv=gkf, verbose=2
        )

        print(
            f"=== Starting GridSearchCV for window={window} (This may take a while...) ==="
        )
        # === 既存のGridSearchCV部分（そのままでOK） ===
        grid_search.fit(X_train, y_train, groups=groups)
        model = grid_search.best_estimator_

        # 一旦全特徴量での評価
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=["普通 (0)", "興奮 (1)"],
            zero_division=0,
            output_dict=True,
        )
        n = len(y_test)
        k = X_train.shape[1]
        log_likelihood = -log_loss(y_test, y_prob, normalize=False)
        AIC = 2 * k - 2 * log_likelihood

        # === 特徴量重要度ランキング ===
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top_features = (
            importances.sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()
        )

        # === 上位N特徴量で再学習 ===
        print(
            f"\n--- Re-training with Top {TOP_N_FEATURES} Features (window={window}) ---"
        )
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        model_top = RandomForestClassifier(
            **grid_search.best_params_,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model_top.fit(X_train_top, y_train)

        y_pred_top = model_top.predict(X_test_top)
        y_prob_top = model_top.predict_proba(X_test_top)
        report_dict_top = classification_report(
            y_test,
            y_pred_top,
            target_names=["普通 (0)", "興奮 (1)"],
            zero_division=0,
            output_dict=True,
        )
        k_top = X_train_top.shape[1]
        log_likelihood_top = -log_loss(y_test, y_prob_top, normalize=False)
        AIC_top = 2 * k_top - 2 * log_likelihood_top

        # === 結果を両方保存 ===
        results_summary.append(
            {
                "window": window,
                "mode": "all_features",
                "num_features": k,
                "best_cv_f1": grid_search.best_score_,
                "test_accuracy": report_dict["accuracy"],
                "test_f1_exciting": report_dict["興奮 (1)"]["f1-score"],
                "test_recall_exciting": report_dict["興奮 (1)"]["recall"],
                "test_precision_exciting": report_dict["興奮 (1)"]["precision"],
                "AIC": AIC,
                "best_params": grid_search.best_params_,
            }
        )

        results_summary.append(
            {
                "window": window,
                "mode": f"top{TOP_N_FEATURES}",
                "num_features": k_top,
                "best_cv_f1": grid_search.best_score_,
                "test_accuracy": report_dict_top["accuracy"],
                "test_f1_exciting": report_dict_top["興奮 (1)"]["f1-score"],
                "test_recall_exciting": report_dict_top["興奮 (1)"]["recall"],
                "test_precision_exciting": report_dict_top["興奮 (1)"]["precision"],
                "AIC": AIC_top,
                "best_params": grid_search.best_params_,
            }
        )

    #   変更点 6: 全てのループ終了後、サマリーを表形式で表示
    print("\n\n=======================================================")
    print("===            All Window Size Results            ===")
    print("=======================================================")

    if not results_summary:
        print("No results to display.")
        return

    # pandas DataFrame にして見やすく表示
    summary_df = pd.DataFrame(results_summary)

    # 表示順を調整
    summary_df = summary_df[
        [
            "window",
            "mode",
            "best_cv_f1",
            "test_accuracy",
            "test_f1_exciting",
            "test_recall_exciting",
            "test_precision_exciting",
            "AIC",
            "num_features",
            "best_params",
        ]
    ]

    # F1やAccuracyを小数点以下4桁で表示
    pd.set_option("display.float_format", "{:.4f}".format)
    # paramsが長すぎると表示が崩れるため、最大幅を設定
    pd.set_option("display.max_colwidth", 50)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
