import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def extract_all_feature_paths(play):
    """
    1つのプレイから、ネストされた dict 構造の特徴パスを再帰的に抽出
    - dict で構成され、すべて bool 値のものを特徴グループとして抽出
    """
    paths = []

    def recurse(d, path):
        for k, v in d.items():
            if isinstance(v, dict):
                if all(isinstance(val, bool) for val in v.values()):
                    paths.append(path + [k])
                recurse(v, path + [k])

    recurse(play, [])
    print(f"パス一覧: {paths}")
    return paths


def build_feature_df_with_context(data, group_paths, window=1):
    """
    指定した特徴パス群（group_paths）をまとめて取り出し、文脈（前後プレイ）も含めた特徴ベクトルに変換
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

    # 文脈（過去/未来プレイ）を結合
    frames = []
    for offset in range(-window, window + 1):
        df_shifted = df_main.shift(offset)
        label = "cur" if offset == 0 else ("prev" if offset < 0 else "next")
        df_shifted.columns = [f"{label}.{col}" for col in df_shifted.columns]
        frames.append(df_shifted)
    df_full = pd.concat(frames, axis=1).fillna(False).astype(int)
    return df_full


def run_isolation_forest(df, contamination=0.05):
    if df.empty or df.shape[1] == 0:
        return None, None
    X = df.fillna(False).astype(int)
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(X)
    scores = clf.decision_function(X)
    return preds, scores


def main():
    # 元データ読み込み
    with open("data/molded_data/777398_molded_data.json", encoding="utf-8") as f:
        data = json.load(f)

    # 全プレイ抽出
    all_plays = []
    for minute in data["minutes"].values():
        all_plays.extend(minute)

    # 最初のプレイから特徴パスを抽出
    if not all_plays:
        raise ValueError("プレイデータが空です")
    group_paths = extract_all_feature_paths(all_plays[0])

    # 特徴パスを保存
    with open("feature_paths.json", "w", encoding="utf-8") as f:
        json.dump(group_paths, f, ensure_ascii=False, indent=2)

    # 文脈付き特徴量を生成
    df = build_feature_df_with_context(all_plays, group_paths, window=1)

    # 異常検知
    preds, scores = run_isolation_forest(df)

    # 結果保存
    results = {
        "grouped_context_features": {
            "predictions": preds.tolist(),
            "scores": scores.tolist(),
            "features": df.columns.tolist()
        }
    }

    with open("isolation_forest_grouped_context.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
