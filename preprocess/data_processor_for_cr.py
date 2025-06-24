import json

# def data_download():
#     with open("data/processed/777471_processed_data.json", encoding="utf-8") as f:
#         game_data = json.load(f)
        
#     return game_data

def get_score(data,gamepk):
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
                    
    score["gamepk"] = gamepk
    score["time"] = time
    score["ex_base_hit_cnt"] = ex_base_hit_cnt
    score["total_score"] = total_score
    score["diff_score"] = diff_score
    score["lead_change_cnt"] = lead_change_cnt
    
    return score
    
def data_process_for_cr(process_data,gamepk):
    score = get_score(process_data,gamepk)
    return score
