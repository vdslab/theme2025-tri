import json

from analysys.data_collection.time_data_sellecting import parse_time

# def data_download():
#     with open("data/processed/777471_processed_data.json", encoding="utf-8") as f:
#         game_data = json.load(f)
        
#     return game_data

def get_score(raw_data,data,gamepk):
    score = {}
    
    # 試合時間
    time = 0
    # 長打数(2ベース以上)
    ex_base_hit_cnt = 0
    # 両チームの合計得点
    total_score = 0
    # 得点差
    diff_score = 0
    # リードチェンジ回数
    lead_team = None
    
    # 日付
    date = raw_data.get("gameData", {}).get("datetime", {}).get("officialDate")
    print(date)
    # team
    team = {}
    team["away"] = raw_data.get("gameData", {}).get("teams", {}).get("away", {}).get("name", "undefined")
    team["home"] = raw_data.get("gameData", {}).get("teams", {}).get("home", {}).get("name", "undefined")

    lead_change_cnt = 0
    for _,play in data.items():
        for _,event in play.items():
            time += event["time"]["diff_time"]/60

            if event["event_type"] in ["double" , "triple" , "home_run"]:
                ex_base_hit_cnt += 1
            
            total_score = event["team_score"]["away"]["pos_score"] + event["team_score"]["home"]["pos_score"]
            
            diff_score = abs(event["team_score"]["away"]["pos_score"] - event["team_score"]["home"]["pos_score"])
        
            if lead_team == "away" and event["team_score"]["away"]["pos_score"] < event["team_score"]["home"]["pos_score"]:
                lead_change_cnt += 1
                
            if lead_team == "home" and event["team_score"]["away"]["pos_score"] > event["team_score"]["home"]["pos_score"]:
                lead_change_cnt += 1
                
            if event["team_score"]["away"]["pos_score"] > event["team_score"]["home"]["pos_score"]:
                lead_team = "away"
            elif event["team_score"]["away"]["pos_score"] < event["team_score"]["home"]["pos_score"]:
                lead_team = "home"
                
        # 全打席を走査して、最小・最大時間を取得
        all_start_times = []
        all_end_times = []
        
        for at_bat_id, event_group in data.items():
            for event_id, event in event_group.items():
                start = parse_time(event["time"]["start_time"])
                end = parse_time(event["time"]["end_time"])
                all_start_times.append(start)
                all_end_times.append(end)
                
        start_time = min(all_start_times)
        end_time = max(all_end_times)
        total_duration = (end_time - start_time).total_seconds()
                    
    score["gamepk"] = gamepk
    score["time"] = total_duration
    score["ex_base_hit_cnt"] = ex_base_hit_cnt
    score["total_score"] = total_score
    score["diff_score"] = diff_score
    score["lead_change_cnt"] = lead_change_cnt
    score["date"] = date
    score["team"] = team
    
    return score
    
def data_process_for_cr(raw_data,process_data,gamepk):
    score = get_score(raw_data,process_data,gamepk)
    return score
