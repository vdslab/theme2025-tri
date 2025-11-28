import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import os

# prev5分を最初の分でパディングするバージョン

# --- 訓練用試合リスト ---
PK_TRAIN = [
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


# いじっていい場所
# ===================================================================

# 推論対象のgamepk
# webの訪問先のgamepkに対応するよう変えて
PK_INFERENCE_TARGET = 778062

# 推論結果の出力先のディレクトリ
# データベースに格納するときはここを適宜変更して
OUTPUT_DIR = "data/catboost_predictions/"

# molded_dataのパスのテンプレート
# web上のパスに合わせて変更して
# 今はtest3って名前にしてる、testと混同するから変えて(懇願)
molded_path_template = "data/test_molded_data/{gamepk}_test3_molded_data.json"

# 推論結果の出力時のファイル名テンプレート
# web上のパスに合わせて変更して
output_filename_template = "prob_{gamepk}.json"

# アノテーションデータが格納されているのパスのテンプレートとファイル名の指定
# web上のパスに合わせて変更して

annotation_directory = "data/anotation_data/"
annotation_filename_template = "cluster_3.csv"


# ===================================================================
# いじっていい場所終わり


# --- データ読み込み対象リスト (訓練用と推論用を結合) ---
PK_LIST_ALL = PK_TRAIN + [PK_INFERENCE_TARGET]


# --- パラメータを、window = 5, all_features,depth = 5, iterations = 500, 12_reaf_reg = 3, kearning_rate = 0.01 に固定 ---
TARGET_WINDOW = 5
FIXED_PARAMS = {
    "depth": 6,
    "iterations": 500,
    "l2_leaf_reg": 3,
    "learning_rate": 0.01,
}


# --- 特徴量を抽出する関数 ---
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

    # データがない場合は空のDataFrameを返す
    if df_main.empty:
        return pd.DataFrame()

    # 試合の最初の分のデータを行として取得
    first_minute_data = df_main.iloc[0]

    for offset in range(-window, 1):
        df_shifted = df_main.shift(offset)

        if offset < 0:
            # 過去のデータを参照する場合
            num_nan_rows = abs(offset)

            # パディング用のDataFrameを作成
            # インデックスをdf_shiftedの先頭 num_nan_rows に合わせる
            padding_df = pd.DataFrame(
                [first_minute_data.values] * num_nan_rows,
                index=df_shifted.index[:num_nan_rows],
                columns=df_shifted.columns,
            )

            # df_shiftedのNaN部分をpadding_dfで埋める
            df_shifted = padding_df.combine_first(df_shifted)
            # ======================================

        label = f"prev{abs(offset)}" if offset < 0 else "cur"
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    return pd.concat(frames, axis=1)


# --- データ読み込みと前処理 ---
def load_and_preprocess(pk_list, annotation_df, window, target_molded_data):
    all_features_list, all_labels_list = [], []
    inference_plays = None

    for gamepk in pk_list:
        gamepk_str = str(gamepk)
        molded_path = molded_path_template.format(gamepk=gamepk_str)

        if gamepk != PK_INFERENCE_TARGET or target_molded_data is None:
            try:
                with open(molded_path, encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"[{gamepk_str}] [SKIP] molded_data not found.")
                continue
        else:
            data = target_molded_data

        all_plays = [play for minute in data["minutes"].values() for play in minute]
        if not all_plays:
            continue

        # 推論対象試合のall_playsを保存
        if gamepk == PK_INFERENCE_TARGET:
            inference_plays = all_plays

        # --- ラベルの存在をチェック ---
        has_annotation = gamepk_str in annotation_df.columns

        # 特徴量抽出
        try:
            group_paths = extract_all_feature_paths(all_plays[0])
        except IndexError:
            print(f"[{gamepk_str}] [INFO] No plays data found.")
            continue

        features_df = build_feature_df_with_context(
            all_plays, group_paths, window=window
        )

        # ラベルの準備と訓練と推論用データの分離
        if gamepk in PK_TRAIN:
            if not has_annotation:
                print(f"[{gamepk_str}] [SKIP] Train game but annotation not found.")
                continue
            labels_series = annotation_df.set_index("gamePK")[gamepk_str]
        elif gamepk == PK_INFERENCE_TARGET:
            # 推論対象の試合はラベルなし、nanにする
            labels_series = pd.Series(np.nan, index=features_df.index)
        else:
            # PK_TRAINにもPK_INFERENCE_TARGETにも含まれない試合は処理しない
            continue

        combined_df = features_df.copy()
        combined_df["label"] = labels_series
        combined_df["gamepk"] = gamepk

        # NaN行を削除
        combined_df.dropna(subset=features_df.columns, inplace=True)

        if combined_df.empty:
            print(f"[{gamepk_str}] [INFO] No valid data found after window processing.")
            continue

        # 特徴量とラベルを分離
        labels_final = combined_df.pop("label")
        features_final = combined_df

        all_features_list.append(features_final)
        all_labels_list.append(labels_final)

    if not all_features_list:
        return None, None, None

    X_full = pd.concat(all_features_list)
    y_full = pd.concat(all_labels_list)

    return X_full, y_full, inference_plays


# --- メイン処理 ---
def catBoost_info_jufge(target_molded_data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(
        f"=== CatBoost Model: Training on {len(PK_TRAIN)} Fixed Games, Inferring on {PK_INFERENCE_TARGET} ==="
    )

    annotation_path = annotation_directory + annotation_filename_template
    try:
        annotation_df = pd.read_csv(annotation_path)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}")
        return

    # 訓練用試合と推論用試合を読み込み、前処理
    X_full, y_full, inference_plays = load_and_preprocess(
        PK_LIST_ALL, annotation_df, TARGET_WINDOW, target_molded_data
    )

    if X_full is None:
        print("[STOP] No processable data found.")
        return

    # 訓練セットを抽出
    train_mask = X_full["gamepk"].isin(PK_TRAIN)
    X_train = X_full[train_mask].drop(columns=["gamepk"])
    y_train = y_full[train_mask].astype(int)

    # 推論セットを抽出
    inference_mask = X_full["gamepk"] == PK_INFERENCE_TARGET
    X_inference_target = X_full[inference_mask].drop(columns=["gamepk"])

    if X_train.empty:
        print("[STOP] Train set is empty. Cannot train model.")
        return
    if X_inference_target.empty:
        print(
            f"[STOP] Inference target ({PK_INFERENCE_TARGET}) has no valid data after preprocessing."
        )
        return

    print(
        f"\n=> Training Final CatBoost Model (Plays: {len(X_train)}, Features: {X_train.shape[1]})..."
    )
    # モデル初期化
    final_model = CatBoostClassifier(
        **FIXED_PARAMS,
        random_seed=42,
        verbose=0,
        class_weights={0: 1, 1: 2},
    )

    # 訓練セットで学習
    final_model.fit(X_train, y_train)
    print(f"   - Model trained on {len(PK_TRAIN)} fixed games.")

    # 推論対象の試合に対して確率を予測
    print(f"\n=> Predicting Probabilities for gamepk {PK_INFERENCE_TARGET}...")
    y_prob_inference = final_model.predict_proba(X_inference_target)

    # 確率をデータフレームに格納
    # 元のインデックスと確率を結合
    prob_df = pd.DataFrame(
        {
            "prob_exciting": y_prob_inference[:, 1],
        },
        index=X_inference_target.index,
    )

    # 推論結果をJSONファイルに出力
    print("\n=> Exporting Inference Probabilities...")

    # prob_dfのインデックスを使って対応するp_idとe_idを取得
    pk_results = {
        "id": PK_INFERENCE_TARGET,
        "data": [
            {
                "x": i,
                "y": int(prob * 100),
                "e_id": inference_plays[idx]["detail"]["e_id"],
                "p_id": inference_plays[idx]["detail"]["p_id"],
                "inning": inference_plays[idx]["detail"]["inning"],
                "inning_top": inference_plays[idx]["detail"]["inning_top"],
            }
            for i, (idx, prob) in enumerate(prob_df["prob_exciting"].items())
        ],
    }

    if pk_results:
        output_filename = os.path.join(
            OUTPUT_DIR, output_filename_template.format(gamepk=PK_INFERENCE_TARGET)
        )
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(pk_results, f, ensure_ascii=False, indent=4)
        print(
            f"   - Successfully saved {len(pk_results)} probabilities to {output_filename}"
        )
    else:
        print(
            f"   - [ERROR] Inference data for gamepk {PK_INFERENCE_TARGET} was empty."
        )

    return pk_results


if __name__ == "__main__":
    target_molded_data = None
    catBoost_info_jufge(target_molded_data)
