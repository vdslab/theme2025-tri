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
    
    # 総イベント時間
    total_event_duration = sum((end - start).total_seconds() for start, end in zip(all_start_times, all_end_times))
    # 10の位で四捨五入
    total_event_duration = round(total_event_duration / 10) * 10

    # 総イベント時間を100分割
    segment_duration = total_event_duration / 100
    
    # 各セグメントに該当するイベントを記録
    segment_events = defaultdict(list)
    
    # イベントを時間順に並べ替え
    events.sort(key=lambda x: x["start"])
    
    # 累積時間を追跡
    cumulative_time = 0
    
    for event in events:
        event_duration = (event["end"] - event["start"]).total_seconds()
        
        # イベントが属するセグメントを決定
        start_segment = int(cumulative_time / segment_duration)
        end_segment = int((cumulative_time + event_duration) / segment_duration)
        
        # イベントが複数のセグメントにまたがる場合
        for segment in range(start_segment, min(end_segment + 1, 100)):
            segment_events[segment].append({
                "play_features": event["play_features"],
                "situation_features": event["situation_features"],
                "detail": event["detail"]
            })
        
        cumulative_time += event_duration

    # --- 出力用に整形 ---
    output = {
        "total_duration_minutes": int(total_duration // 60),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "minutes": {},
    }

    for segment, features in segment_events.items():
        output["minutes"][str(segment)] = features

    # --- 保存 ---
    with open(f"data/molded_data/{gamepk}_molded_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"保存完了：{gamepk}_molded_data.json")
    print(f"総イベント時間: {total_event_duration:.2f}秒")
    print(f"セグメント時間: {segment_duration:.2f}秒")
    
    return output
