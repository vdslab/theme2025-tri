import json

with open("data/raw/game/777708.json", encoding="utf-8") as f:
    game_data = json.load(f)

with open("master/bb_evaluation_master.json", encoding="utf-8") as f:
    evaluation_master = json.load(f)

play_features = evaluation_master["play_features"]
situation_features = evaluation_master["situation_features"]

all_plays = game_data["liveData"]["plays"]["allPlays"]

def get_situation_score(event,play,isLast,a,b):
    
    inning = play["about"].get("inning",{})
    
    # inning_scoreを算出
    inning_score = 0
    if inning <= 3:
        inning_score = situation_features["inning_phase"]["early"]
    elif inning <= 6:
        inning_score = situation_features["inning_phase"]["middle"]
    else:
        inning_score = situation_features["inning_phase"]["late"]
    
    a_score = play["result"].get("awayScore",{})
    h_score = play["result"].get("awayScore",{})
    
    isTopInning = play["about"].get("isTopInning",{})
    
    rbi = play["result"].get("awayScore",{})
    
    # 相対的な得点差
    # pre_diff_score,diff_scoreを算出
    pre_diff_score = 0
    diff_score = 0
    
    if isLast:
        if isTopInning:
            pre_a_score = a_score - rbi
            pre_diff_score = func_score_diff(pre_a_score - h_score)
            diff_score = func_score_diff(a_score - h_score)
        else:
            pre_h_score = h_score - rbi
            pre_diff_score = func_score_diff(pre_h_score - a_score)
            diff_score = func_score_diff(h_score - a_score)
    
    else:
        if isTopInning:
            pre_diff_score = func_score_diff(a_score - h_score)
            diff_score = func_score_diff(a_score - h_score)
        else:
            pre_diff_score = func_score_diff(a_score - h_score)
            diff_score = func_score_diff(a_score - h_score)
            
    # 逆転のランナーがいるか
    # 負けている時に限る
    # 点差よりランナー数が多いならtrue
    
    # ランナー状況
    
    return a * (inning_score + pre_diff_score) + b * (inning_score + diff_score)

def func_score_diff(score_diff):
    diff_score = 0
    if score_diff <= -3:
        diff_score = situation_features["score_difference"]["minus_less_3"]   
    elif  score_diff == -2:
        diff_score = situation_features["score_difference"]["minus_2"]
    elif  score_diff == -1:
        diff_score = situation_features["score_difference"]["minus_1"]
    elif  score_diff == 0:
        diff_score = situation_features["score_difference"]["tie"]
    elif  score_diff == 1:
        diff_score = situation_features["score_difference"]["plus_1"] 
    elif  score_diff == 2:
        diff_score = situation_features["score_difference"]["plus_2"] 
    elif  score_diff >= 3:
        diff_score = situation_features["score_difference"]["plus_more_3"] 
    
    return diff_score

def get_play_event_score(event,play,isLast):
    score = 0
    result = play["result"]
    eventType = result.get("eventType")
    unique_score = 0
    
    if eventType == "home_run":
        unique_score += play_features["hit_event"]["home_run"]
    elif eventType == "triple":
        unique_score += play_features["hit_event"]["triple"]
    elif eventType == "double":
        unique_score += play_features["hit_event"]["double"]
    elif eventType == "single":
        unique_score += play_features["hit_event"]["single"]
    else:
        unique_score += play_features["hit_event"]["others"]
    
    # NOTE:playのresultが上記のものである場合、playEventsの最後のeventに関しては上記のunique_scoreを適用させる
    # 上記に含まれない場合は、playEventsの "description" を基にscoreを算出する
        
    # TODO:descriptionの値によってscoreを変える可能性はある
    # 現状0.2で固定
    # event["detail"]["description"]
    score = 0.2
    
    # NOTE:最終要素の場合のみユニークスコアに変換
    if isLast:
        score = unique_score
        
    # 打点の特徴に対する評価
    # NOTE:負けているときdiff_scoreは負
    
    rbi = result.get("rbi")
    a_score = result.get("awayScore",{})
    h_score = result.get("awayScore",{})

    isTopInning = play["about"].get("isTopInning",{})
    if isTopInning:
        pre_a_score = a_score-rbi
        diff_score = pre_a_score-h_score
    else:
        pre_h_score = h_score-rbi
        diff_score = pre_h_score-a_score
    
    rbi_score = 0
    unique_rbi_score = 0
    if diff_score == 0:
        if rbi > 0:
            unique_rbi_score = play_features["rbi_impact"]["go_ahead_rbi"]
    elif diff_score < 0:
        if rbi == diff_score*(-1):
            unique_rbi_score = play_features["rbi_impact"]["tie_rbi"]
        elif rbi > diff_score*(-1):
            unique_rbi_score = play_features["rbi_impact"]["sayonara_rbi"]
    else:
        if rbi > 0:
            unique_rbi_score = play_features["rbi_impact"]["regular_rbi"]
        
    # NOTE:最終要素の場合のみユニークスコアに変換
    if isLast:
        rbi_score = unique_rbi_score
        
    return score + rbi_score

def get_scores():
    scores = []
    for play in all_plays:
        # NOTE:True:表,False:裏
        if  play["about"]["isTopInning"] == False:
            events = play["playEvents"]

            for idx,event in enumerate(events):
                score = 0
                # NOTE:周辺イベントの排除(ウォーミングアップやタイム)
                if event["type"] == "action":
                    continue
                
                isLast = idx == len(events)-1
                
                score = get_play_event_score(event,play,isLast) + get_situation_score(event,play,isLast,a = 0.5,b = 0.5)
                scores.append(score)
    return scores