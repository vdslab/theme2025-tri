import json

def data_download():
    with open("data/raw/game/777490.json", encoding="utf-8") as f:
        game_data = json.load(f)
    return game_data

def process_data(data):
    allPlays = data["liveData"]["plays"]["allPlays"]
    event_lookup = {}
    isInningTop_ = False
    pre_runner_state = {}
    pre_home_score,pre_away_score,pos_home_score,pos_away_score = 0,0,0,0
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
            
            event_lookup[p_idx][e_idx], pre_runner_state,pre_away_score,pre_home_score = process_event(play,event,is_inning_first,isPlayFirst,isLast,pre_runner_state,p_idx,e_idx,pre_home_score,pre_away_score,pos_home_score,pos_away_score)

                
    return event_lookup
    
def process_event(play,event,is_inning_first,isPlayFirst,isLast,pre_runner_state,p_idx,e_idx,pre_home_score,pre_away_score,pos_home_score,pos_away_score):
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
    for runner in runners:
        start_ = None
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
        
    
    processed_event["is_away"] = is_away
    processed_event["is_inning_first"] = is_inning_first
    processed_event["inning"] = inning
    processed_event["event_type"] = event_type
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
    
    return processed_event, pre_runner_state,pos_away_score,pos_home_score

def output_data(processed_data):
    output_path = "data/processed/777490_processed_data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

data = data_download()
processed_data = process_data(data)
output_data(processed_data)