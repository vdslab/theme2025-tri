import json
import logging
logging.basicConfig(level=logging.WARNING)

from calculate.measure_time import calc_time_diff

with open("data/raw/game/777708.json", encoding="utf-8") as f:
    data = json.load(f)


all_plays = data["liveData"]["plays"]["allPlays"]

# NOTE:タグ確認用
# keys = set()
# for play in all_plays:
#     keys.update(play.keys())
    
# print(keys)

for play in all_plays:
    # NOTE:rue:表,False:裏
    if  play["about"]["isTopInning"] == True:
        print("name : " + play["matchup"]["batter"]["fullName"])
        events = play["playEvents"]

        for event in events:
            # NOTE:周辺イベントの排除(ウォーミングアップやタイム)
            if event["type"] == "action":
                continue
            
            calc_time_diff(event["startTime"] ,event["endTime"])
            try:
                print(event["details"]["description"])
                
            # NOTE:keyがない:KeyError,eventが辞書でない等:TypeError
            except (KeyError, TypeError) as e:
                logging.warning("description 取得失敗 (%s): %s", e, event)
                
