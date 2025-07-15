import json
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ^^^ gamepkの指定 ---
gamepk = "777866"
# --- JSONファイルの読み込み ---
with open(f"data/processed_for_ra/{gamepk}_processed_for_ra_data.json", encoding="utf-8") as f:
    match_data = json.load(f)

# --- 開始・終了時刻の取得 ---
def parse_time(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))

def time_data_sellecting(gamepk,match_data):
    # 全打席を走査して、最小・最大時間を取得
    all_start_times = []
    all_end_times = []
    events = []
    for at_bat_id, event_group in match_data.items():
        for event_id, event in event_group.items():
            start = parse_time(event["time"]["start_time"])
            end = parse_time(event["time"]["end_time"])
            all_start_times.append(start)
            all_end_times.append(end)
            events.append({
                "start": start,
                "end": end,
                "play_features": event["play_features"],
                "situation_features": event["situation_features"],
                "detail": event["detail"]
            })

    start_time = min(all_start_times)
    end_time = max(all_end_times)
    total_duration = (end_time - start_time).total_seconds()

    # --- 1分単位に分割し、該当イベントを記録 ---
    minute_events = defaultdict(list)
    for event in events:
        start_minute = int((event["start"] - start_time).total_seconds() // 60)
        end_minute = int((event["end"] - start_time).total_seconds() // 60)
        for minute in range(start_minute, end_minute + 1):
            minute_events[minute].append({
                "play_features": event["play_features"],
                "situation_features": event["situation_features"],
                "detail": event["detail"]
            })

    # --- 出力用に整形 ---
    output = {
        "total_duration_minutes": int(total_duration // 60),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "minutes": {},
    }

    for minute, features in minute_events.items():
        output["minutes"][str(minute)] = features

    # --- 保存 ---
    # with open(f"data/molded_data/{gamepk}_molded_data.json", "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"保存完了：{gamepk}_molded_data.json")
    
    return output
