import json
import requests
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.calculate.measure_time import calc_time_diff

def data_download(gamepk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{gamepk}/feed/live"
    resp = requests.get(url)
    if resp.ok:
        return resp.json()
    else:
        return None

def process_data(data):
    allPlays = data["liveData"]["plays"]["allPlays"]
    event_lookup = {}
    isInningTop_ = False
    pre_runner_state = {}
    pre_home_score,pre_away_score,pos_home_score,pos_away_score = 0,0,0,0
    last_inning = max(play["about"]["inning"] for play in allPlays)
    for p_idx, play in enumerate(allPlays):
        playEvents = play["playEvents"]
        isInningTop = play["about"]["isTopInning"]
        
        event_lookup[p_idx] = {}
        for e_idx, event in enumerate(playEvents):
            
            is_inning_first = isInningTop_ != isInningTop
            if is_inning_first:
                isInningTop_ = not isInningTop_
            
            isLast = e_idx == len(play["playEvents"])-1
            isPlayFirst = p_idx == 0
            
            event_lookup[p_idx][e_idx], pre_runner_state,pre_away_score,pre_home_score = process_event(play,event,is_inning_first,isPlayFirst,isLast,pre_runner_state,p_idx,e_idx,pre_home_score,pre_away_score,pos_home_score,pos_away_score,last_inning)
                
    return event_lookup
    
def process_event(play,event,is_inning_first,isPlayFirst,isLast,pre_runner_state,p_idx,e_idx,pre_home_score,pre_away_score,pos_home_score,pos_away_score,last_inning):
    processed_event = {}
    
    # is away
    is_away = None
    if play["about"]["isTopInning"] == True:
        is_away = True
    else:
        is_away = False
    
    # inning
    inning = play["about"]["inning"]
    
    # event type
    event_type = event["type"]
    pe_type = play["result"]["eventType"]
    
    # is base running play
    is_base_running_play = event.get("isBaseRunningPlay", "null")
    if isLast:
        event_type = pe_type
        
    batter = {
        "id":play["matchup"]["batter"]["id"],
        "full_name":play["matchup"]["batter"]["fullName"],
    }
    
    # runner state
    runner_state = {}
    pos_runner_state = {}
    if is_inning_first:
        pre_runner_state = {
            "1B": {
                "id":None,
                "full_name":None
            },
            "2B": {
                "id":None,
                "full_name":None
            },
            "3B": {
                "id":None,
                "full_name":None
            },
        }

    base_movements = {}
    
    runners = play["runners"]
    start_ = None
    for runner in runners:
        if runner["details"]["playIndex"] == event["index"]:

            origin = runner["movement"]["originBase"]
            start = runner["movement"]["start"]
            end = runner["movement"]["end"]

            if origin not in base_movements:
                base_movements[start] = end
                start_ = start
            else:
                base_movements[start_] = end
    
    pos_runner_state = {
        "1B": {"id": None, "full_name": None},
        "2B": {"id": None, "full_name": None},
        "3B": {"id": None, "full_name": None},
    }

    state = {"1B":False,"2B":False,"3B":False}
    for k,v in base_movements.items():
        state[k] = True
        if not (v == "score"):
            if k == None:
                if v != None:
                    pos_runner_state[v] = batter
            else:
                pos_runner_state[v] = pre_runner_state[k]
    for sk,sv in state.items():
        if sv == False and pre_runner_state[sk]["id"] != None:
            pos_runner_state[sk] = pre_runner_state[sk]
    
    # runner count
    runner_count = {}
    pre_runner_count = 0
    for v in pre_runner_state.values():
        if v["id"] != None:
            pre_runner_count += 1
    
    pos_runner_count = 0
    for v in pos_runner_state.values():
        if v["id"] != None:
            pos_runner_count += 1
            
    # score from event
    score_from_event = 0
    for v in base_movements.values():
        if v == "score":
            score_from_event += 1
            
    # team score
    team_score = {}
    away = {}
    home = {}
    if is_away == True:
        pos_away_score = pre_away_score + score_from_event
        pos_home_score = pre_home_score
    else:
        pos_away_score = pre_away_score
        pos_home_score = pre_home_score + score_from_event
        
    # rbi
    rbi = 0
    if isLast:
        rbi = play["result"]["rbi"]
    
    # is_last_inning    
    is_last_inning =  is_away == False and inning == last_inning
    
    # time
    time = {}
    start_time = event.get("startTime",{})
    end_time = event.get("endTime",{})
    diff_time = calc_time_diff(start_time,end_time)
    
    processed_event["is_away"] = is_away
    processed_event["is_inning_first"] = is_inning_first
    processed_event["inning"] = inning
    processed_event["event_type"] = event_type
    processed_event["is_base_running_play"] = is_base_running_play
    processed_event["batter"] = batter
    processed_event["runner_state"] = runner_state
    runner_state["pre_runner_state"] = pre_runner_state
    runner_state["pos_runner_state"] = pos_runner_state
    processed_event["runner_count"] = runner_count
    runner_count["pre_runner_count"] = pre_runner_count
    runner_count["pos_runner_count"] = pos_runner_count
    processed_event["score_from_event"] = score_from_event
    pre_runner_state = pos_runner_state
    team_score["away"] = away
    team_score["home"] = home
    away["pre_score"] = pre_away_score
    away["pos_score"] = pos_away_score
    home["pre_score"] = pre_home_score
    home["pos_score"] = pos_home_score
    processed_event["team_score"] = team_score
    processed_event["rbi"] = rbi
    processed_event["is_last_inning"] = is_last_inning
    processed_event["time"] = time
    time["start_time"] = start_time
    time["end_time"] = end_time
    time["diff_time"] = diff_time
    
    return processed_event, pre_runner_state,pos_away_score,pos_home_score

def output_data(processed_data,gamepk):
    output_path = f"data/processed/{gamepk}_processed_data.json"

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(processed_data, f, ensure_ascii=False, indent=4)

def data_process(gamepk):
    raw_data = data_download(gamepk)
    processed_data = process_data(raw_data)
    output_data(processed_data,gamepk)
    
    return raw_data,processed_data
