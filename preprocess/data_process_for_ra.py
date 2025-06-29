import json
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_scores(processed_data):
    game_data = processed_data
    scores_data = {}
    for p_idx,play in  game_data.items():
        p_score = {}
        for e_idx,event in play.items():
            if event["event_type"] == "action" and event["is_base_running_play"] == None:
                continue
            e_score = {}
            play_features = {}
            situation_features = {}
            
            get_play_score(event,play_features)
            get_situation_score(event,situation_features)
            
            p_score[e_idx] = e_score
            e_score["play_features"] = play_features
            e_score["situation_features"] = situation_features
            e_score["time"] = event["time"]
            e_score["detail"] = event["detail"]
            
        scores_data[p_idx] = p_score
    return scores_data
        
def get_play_score(event,play_features):
    # hit_event
    hit_event = {}
    hit_event["single"] = event["event_type"] == "single"
    hit_event["double"] = event["event_type"] == "double"
    hit_event["triple"] = event["event_type"] == "triple"
    hit_event["home_run"] = event["event_type"] == "home_run"
    
    # rbi_impact
    rbi_impact = {}
    rbi = event["rbi"]
    pre_score_diff = 0
    team_score = event["team_score"]
    rbi_impact["regular_rbi"] = False
    rbi_impact["tie_rbi"] = False
    rbi_impact["go_ahead_rbi"] = False
    rbi_impact["sayonara_rbi"] = False
    # TODO:キモすぎるのでいつか直す
    if rbi > 0:
        if event["is_away"]:
            if team_score["home"]["pre_score"] - team_score["away"]["pre_score"] >= 0:
                pre_score_diff = team_score["home"]["pre_score"] - team_score["away"]["pre_score"]
                if pre_score_diff == rbi:
                    rbi_impact["tie_rbi"] = True
                elif pre_score_diff > rbi:
                    if event["is_last_inning"]:
                        rbi_impact["sayonara_rbi"] = True
                    else:
                        rbi_impact["go_ahead_rbi"] = True
                else:
                    rbi_impact["regular_rbi"] = True
            else:
                rbi_impact["regular_rbi"] = True
                        
        else:
            if team_score["away"]["pre_score"] - team_score["home"]["pre_score"] > 0:
                pre_score_diff = team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
                if pre_score_diff == rbi:
                    rbi_impact["regular_rbi"] = True
                elif pre_score_diff > rbi:
                    if event["is_last_inning"]:
                        rbi_impact["sayonara_rbi"] = True
                    else:
                        rbi_impact["go_ahead_rbi"] = True
                else:
                    rbi_impact["regular_rbi"] = True
            else:
                rbi_impact["regular_rbi"] = True
                
    play_features["hit_event"] = hit_event
    play_features["rbi_impact"] = rbi_impact

def get_situation_score(event,situation_features):
    # runner_status
    runner_status = {}
    runner_state = event["runner_state"]
    is_runner = {
        "1B":False,
        "2B":False,
        "3B":False
    }
    # NOTE:イベント前はpre,イベント後はpos
    for k,v in runner_state["pre_runner_state"].items():
        if v["id"]:
            is_runner[k] = True
        
    runner_status["none"] = not is_runner["1B"] and not is_runner["2B"] and not is_runner["3B"]
    runner_status["first"] = is_runner["1B"] and not is_runner["2B"] and not is_runner["3B"]
    runner_status["second"] = not is_runner["1B"] and is_runner["2B"] and not is_runner["3B"]
    runner_status["third"] = not is_runner["1B"] and not is_runner["2B"] and is_runner["3B"]
    runner_status["first-second"] = is_runner["1B"] and is_runner["2B"] and not is_runner["3B"]
    runner_status["first-third"] = is_runner["1B"] and not is_runner["2B"] and is_runner["3B"]
    runner_status["second-third"] = not is_runner["1B"] and is_runner["2B"] and is_runner["3B"]
    runner_status["first-second-third"] = is_runner["1B"] and is_runner["2B"] and is_runner["3B"]
    
    # is_goahead_runner_on_base
    is_goahead_runner_on_base = False
    pre_runner_count = event["runner_count"]["pre_runner_count"]
    team_score = event["team_score"]

    # NOTE:イベント前はpre,イベント後はpos
    pre_score_diff = 0
    if event["is_away"]:
        # NOTE:イベント前はpre,イベント後はpos
        pre_score_diff = team_score["home"]["pre_score"] - team_score["away"]["pre_score"]
        if pre_score_diff > 0 and pre_runner_count > pre_score_diff:
            is_goahead_runner_on_base = True
    else:
        # NOTE:イベント前はpre,イベント後はpos
        pre_score_diff = team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
        if pre_score_diff > 0 and pre_runner_count > pre_score_diff:
            is_goahead_runner_on_base = True
    
    # score_difference
    score_difference = {}
    if event["is_away"]:
        # NOTE:イベント前はpre,イベント後はpos
        pre_score_diff = team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
        
        score_difference["minus_less_3"] = pre_score_diff <= -3
        score_difference["minus_2"] = pre_score_diff == -2
        score_difference["minus_1"] = pre_score_diff == -1
        score_difference["tie"] = pre_score_diff == 0
        score_difference["plus_1"] = pre_score_diff == 1
        score_difference["plus_2"] = pre_score_diff == 2
        score_difference["plus_more_3"] = pre_score_diff >= 3
    
    # inning_phase
    inning_phase = {}
    inning_phase["early"] = event["inning"] <= 3
    inning_phase["early"] = event["inning"] <= 6
    inning_phase["early"] = event["inning"] >= 7
    
    situation_features["runner_status"] = runner_status
    situation_features["is_goahead_runner_on_base"] = is_goahead_runner_on_base
    situation_features["score_difference"] = score_difference
    situation_features["inning_phase"] = inning_phase

def output_data(scores_data,gamepk):
    output_path = f"data/processed_for_ra/{gamepk}_processed_for_ra_data.json"

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(scores_data, f, ensure_ascii=False, indent=4)
        
        
def data_process_for_ra(processed_data,gamepk):
    scores_data = get_scores(processed_data)
    output_data(scores_data,gamepk)
    
    return scores_data
