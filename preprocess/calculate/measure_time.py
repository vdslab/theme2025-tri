from datetime import datetime

# NOTE:ISO 8601フォーマットの文字列（小数秒あり）を引数にとる
def calc_time_diff(start_time,end_time):

    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    duration = end_dt - start_dt
    
    return duration.total_seconds()
