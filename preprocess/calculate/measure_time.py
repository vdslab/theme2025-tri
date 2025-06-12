from datetime import datetime

def calc_time_diff(start_time,end_time):
    # ISO 8601フォーマットの文字列（小数秒あり）
    start_time = "2025-05-31T01:46:07.680Z"
    end_time = "2025-05-31T02:08:48.432Z"

    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    duration = end_dt - start_dt

    print("Duration (seconds):", duration.total_seconds())
