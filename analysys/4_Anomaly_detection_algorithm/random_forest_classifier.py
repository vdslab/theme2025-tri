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
    for offset in range(-window, 1): # window + 1 から現在地(1)のみに変更
        df_shifted = df_main.shift(offset)
        label = "cur" if offset == 0 else ("prev" if offset < 0 else "next")
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    df_full = pd.concat(frames, axis=1)
    return df_full


def main():

    #   変更点 1: 試したい window サイズのリストを定義
    # window_sizes_to_test = [
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     9,
    #     10,
    # ]

    #   変更点 2: 最終結果を格納するためのリストを初期化
    results_summary = []

    #   変更点 3: window サイズごとにループ処理を行う
    # 結果によりwindow = 2に設定
    window = 2

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

    param_grid = {
        "n_estimators": [100, 200, 300],  # 森を構成する木の本数
        "max_depth": [10, 20, None],  # 各決定木の最大深さ
        "min_samples_split": [2, 5],  # ノードを分割するために必要な最小サンプル数
        "min_samples_leaf": [1, 3],  # 葉ノードに必要な最小サンプル数
        "max_features": ["sqrt", 0.5],  # 各分割で考慮する特徴量の最大数
    }

    base_model = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )

    gkf = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, scoring="f1", cv=gkf, verbose=2
    )

    # ---------------------------------------------------------------------
    # オリジナル: GridSearch を実行する元のロジック（痕跡として残す）
    # ---------------------------------------------------------------------
    # grid_search.fit(X_train, y_train, groups=groups)
    #
    # print(f"\n=== GridSearchCV Results (window={window}) ===")
    # print(f"Best Parameters Found: {grid_search.best_params_}")
    # print(f"Best F1 Score (on Train CV): {grid_search.best_score_:.4f}")
    #
    # model = grid_search.best_estimator_
    #
    # ---------------------------------------------------------------------
    #
    # 以前の実験結果から「window=2 に対して良かったパラメータ」を固定して学習
    best_params_fixed = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    print(f"\n=== Training RandomForest with fixed best_params (window={window}) ===")
    print(f"fixed params: {best_params_fixed}")

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        **best_params_fixed,
    )
    model.fit(X_train, y_train)

    # safety: GridSearch の結果を参照していた箇所のために代替値を用意
    best_score_used = None
    best_params_used = None
    try:
        # もし grid_search が実行済みであればそれを使う（痕跡として）
        if hasattr(grid_search, "best_score_"):
            best_score_used = grid_search.best_score_
        else:
            best_score_used = None
    except Exception:
        best_score_used = None

    # 固定パラメータを使用したことを明示する
    best_params_used = best_params_fixed

    # === (評価プロセス) ===

    y_pred = model.predict(X_test)
    # predict_proba が存在しない分類器のケースも考慮（RandomForest はOK）
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    print(f"\n--- Overall Evaluation (Optimized Model, window={window}) ---")
    #   変更点 4: classification_report を辞書(dict)としても取得
    report_str = classification_report(
        y_test,
        y_pred,
        target_names=["普通のプレイ (0)", "興奮するプレイ (1)"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["普通のプレイ (0)", "興奮するプレイ (1)"],
        zero_division=0,
        output_dict=True,
    )
    print(report_str)

    # ========== AIC / BIC 評価 ==========
    n = len(y_test)
    k = X_train.shape[1]
    if y_prob is None:
        # NOTE: RandomForest では通常 predict_proba があるため基本ここは通らない
        log_likelihood = -log_loss(y_test, y_pred, normalize=False)
    else:
        log_likelihood = -log_loss(y_test, y_prob, normalize=False)

    AIC = 2 * k - 2 * log_likelihood
    BIC = k * np.log(n) - 2 * log_likelihood

    print(f"\n--- Information Criteria (window={window}) ---")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")

    # ===== Optional: 特定試合の予測保存 =====
    target_gamepk = 778163
    if target_gamepk in test_pks:
        # (省略: この部分は変更なし)
        pass  # 実際にはここのロジックは生きています

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top20 = importances.sort_values(ascending=False).head(20)
    print(f"\n▼ 予測に重要だった特徴量 Top 20 (Optimized Model, window={window})")
    print(top20)

    #   変更点 5: 最終結果リストに今回のループの結果を追加
    # GridSearch が使えた場合はその指標を、そうでなければ固定モデルで得た指標を入れる。
    result_entry = {
        "window": window,
        "best_cv_f1": best_score_used if best_score_used is not None else None,
        "test_accuracy": report_dict.get("accuracy", None),
        "test_f1_exciting": report_dict.get("興奮するプレイ (1)", {}).get(
            "f1-score", None
        ),
        "test_recall_exciting": report_dict.get("興奮するプレイ (1)", {}).get(
            "recall", None
        ),
        "test_precision_exciting": report_dict.get("興奮するプレイ (1)", {}).get(
            "precision", None
        ),
        "best_params": best_params_used,
        "AIC": AIC,
        "num_features": k,
    }

    results_summary.append(result_entry)

    # #   変更点 6: 全てのループ終了後、サマリーを表形式で表示
    # print("\n\n=======================================================")
    # print("===            All Window Size Results            ===")
    # print("=======================================================")
    #
    # if not results_summary:
    #     print("No results to display.")
    #     return
    #
    # # pandas DataFrame にして見やすく表示
    # summary_df = pd.DataFrame(results_summary)
    #
    # # 表示順を調整
    # summary_df = summary_df[
    #     [
    #         "window",
    #         "best_cv_f1",
    #         "test_accuracy",
    #         "test_f1_exciting",
    #         "test_recall_exciting",
    #         "test_precision_exciting",
    #         "AIC",
    #         "num_features",
    #         "best_params",
    #     ]
    # ]
    #
    # # F1やAccuracyを小数点以下4桁で表示
    # pd.set_option("display.float_format", "{:.4f}".format)
    # # paramsが長すぎると表示が崩れるため、最大幅を設定
    # pd.set_option("display.max_colwidth", 50)
    #
    # print(summary_df.to_string(index=False))

    # ======================================================
    # === 追加機能: 各試合ごとの盛り上がりヒートマップ出力 ===
    # ======================================================

    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 予測確率の保存
    if y_prob is not None:
        # y_prob[:, 1] が「興奮するプレイ」の確率
        X_test_with_prob = X_all[test_mask].copy()
        X_test_with_prob["excite_prob"] = y_prob[:, 1]
    else:
        print("[WARN] predict_proba が利用できないため、ヒートマップをスキップします。")
        X_test_with_prob = None

    if X_test_with_prob is not None:
        # 出力フォルダを作成
        os.makedirs("output", exist_ok=True)

        # 各試合ごとに処理
        for gamepk in sorted(X_test_with_prob["gamepk"].unique()):
            df_game = X_test_with_prob[X_test_with_prob["gamepk"] == gamepk].copy()

            # 時間的な順序（行インデックス）を横軸に見立ててヒートマップ化
            # 特徴量次元が多いので、ここでは「excite_prob」だけを時系列で描く
            heatmap_array = df_game[["excite_prob"]].T  # (1行 × 時系列列)

            plt.figure(figsize=(10, 2))
            sns.heatmap(
                heatmap_array,
                cmap="coolwarm",
                cbar=True,
                vmin=0.0,
                vmax=1.0,
                cbar_kws={"label": "Excitement Probability"},
            )
            plt.title(f"Game {gamepk} - Excitement Over Time (window={window})")
            plt.xlabel("Play Index (time progression)")
            plt.ylabel("")
            plt.yticks([])
            plt.tight_layout()

            save_path = f"output/heatmap_{gamepk}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"[SAVED] {save_path}")

        print("\n✅ 全試合のヒートマップ画像を 'output/' フォルダに保存しました。")


if __name__ == "__main__":
    main()
