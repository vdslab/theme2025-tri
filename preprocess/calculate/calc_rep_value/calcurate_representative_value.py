import json
import os
import glob
import numpy as np
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# --- 対象ディレクトリ（必要に応じて変更） ---
target_dir = "data/LRA_combined_data/LRA_hit_event__score_difference"

# --- JSONファイルを全取得 ---
json_files = glob.glob(os.path.join(target_dir, "*_highlight_predictions_*.json"))

# --- 結果格納用リスト ---
all_probs = []
event_probs = []

# --- 各ファイルから確率を抽出 ---
for file_path in json_files:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        for minute_data in data.values():
            prob = minute_data.get("highlight_probability", None)
            if prob is None:
                continue
            all_probs.append(prob)
            if minute_data.get("event_type") is not None:
                event_probs.append(prob)

# --- 統計出力関数 ---
def summarize(probs, label):
    if not probs:
        print(f"{label}: データが存在しません。")
        return
    probs_array = np.array(probs)
    count_90_or_more = np.sum(probs_array >= 90.0)

    print(f"\n {label} におけるハイライト確率統計:")
    print(f"  最大値   : {np.max(probs_array):.1f}%")
    print(f"  最小値   : {np.min(probs_array):.1f}%")
    print(f"  平均値   : {np.mean(probs_array):.1f}%")
    print(f"  90%以上の個数: {count_90_or_more} 件")

# --- 結果表示 ---
summarize(all_probs, "全体")
summarize(event_probs, "イベントあり（event_type ≠ None）")
