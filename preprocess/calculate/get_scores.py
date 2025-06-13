import json

with open("data/raw/game/777708.json", encoding="utf-8") as f:
    game_data = json.load(f)

with open("master/bb_evaluation_master.json", encoding="utf-8") as f:
    evaluation_master = json.load(f)

play_features = evaluation_master["play_features"]
situation_features = evaluation_master["situation_features"]

all_plays = game_data["liveData"]["plays"]["allPlays"]

def get_situation_score(event):
    return (

    )

def get_before_situation_score(play):
    return get_situation_score("first_only", "true", "tie", "early")

def get_after_situation_score(play):
    return get_situation_score("bases_loaded", "false", "plus_minus_3_or_more", "late")

def play_event_score(event,play,isLast):
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
        
    return score

def get_score(event, a=0.5, b=0.5):
    return play_event_score(event) * (a * get_before_situation_score(play) + b * get_after_situation_score(play))

for play in all_plays:
    # NOTE:True:表,False:裏
    if  play["about"]["isTopInning"] == False:
        events = play["playEvents"]

        for idx,event in enumerate(events):
            # NOTE:周辺イベントの排除(ウォーミングアップやタイム)
            if event["type"] == "action":
                continue
            isLast = len(events)-1
            # TODO:消すこと
            # print(play_event_score(event,play,isLast))
            
    