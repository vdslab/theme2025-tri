import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import GroupKFold, GridSearchCV

# --- 対象試合リスト ---
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


# --- 特徴量抽出関数（既存コードそのまま） ---
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
    for offset in range(-window, 1):
        df_shifted = df_main.shift(offset)
        label = f"prev{abs(offset)}" if offset < 0 else "cur"
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    return pd.concat(frames, axis=1)


# --- メイン処理 ---
def main():
    window_sizes_to_test = [1, 2, 3, 4, 5]
    TOP_N_FEATURES = 50
    results_summary = []

    annotation_path = "data/anotation_data/cluster_3.csv"
    annotation_df = pd.read_csv(annotation_path)

    for window in window_sizes_to_test:
        print(f"\n=== Running CatBoost model with window={window} ===")

        all_features_list, all_labels_list = [], []

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
            if not all_plays or gamepk_str not in annotation_df.columns:
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

        groups = X_all[train_mask]["gamepk"]
        X_train = X_all[train_mask].drop(columns=["gamepk"])
        y_train = y_all[train_mask]
        X_test = X_all[test_mask].drop(columns=["gamepk"])
        y_test = y_all[test_mask]

        if X_train.empty or X_test.empty:
            print(f"[STOP] Train or Test set is empty for window={window}.")
            continue

        # === CatBoost グリッドサーチ用パラメータ ===
        param_grid = {
            "depth": [6, 8, 10],
            "iterations": [500, 1000],
            "learning_rate": [0.01, 0.03, 0.05],
            "l2_leaf_reg": [1, 3, 5],
        }

        base_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=42,
            verbose=0,
            class_weights={0: 1, 1: 2},
        )

        gkf = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1",
            cv=gkf,
            verbose=2,
        )

        grid_search.fit(X_train, y_train, groups=groups)
        model = grid_search.best_estimator_

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

        importances = pd.Series(model.get_feature_importance(), index=X_train.columns)
        top_features = (
            importances.sort_values(ascending=False).head(TOP_N_FEATURES).index.tolist()
        )

        # --- 上位N特徴量で再学習 ---
        model_top = CatBoostClassifier(
            **grid_search.best_params_,
            class_weights={0: 1, 1: 2},
            random_seed=42,
            verbose=0,
        )
        model_top.fit(X_train[top_features], y_train)
        y_pred_top = model_top.predict(X_test[top_features])
        y_prob_top = model_top.predict_proba(X_test[top_features])
        report_dict_top = classification_report(
            y_test,
            y_pred_top,
            target_names=["普通 (0)", "興奮 (1)"],
            zero_division=0,
            output_dict=True,
        )
        k_top = X_train[top_features].shape[1]
        log_likelihood_top = -log_loss(y_test, y_prob_top, normalize=False)
        AIC_top = 2 * k_top - 2 * log_likelihood_top

        # --- 結果記録 ---
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

    # --- 結果表示 ---
    summary_df = pd.DataFrame(results_summary)
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
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_colwidth", 50)
    print("\n=== CatBoost Summary Results ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
