import json
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.WARNING)

from calculate.measure_time import calc_time_diff

with open("data/raw/game/777708.json", encoding="utf-8") as f:
    data = json.load(f)

all_plays = data["liveData"]["plays"]["allPlays"]

durations = []
sec = 0

for play in all_plays:
    # NOTE:True:表,False:裏
    if  play["about"]["isTopInning"] == True:
        # print("name : " + play["matchup"]["batter"]["fullName"])
        events = play["playEvents"]

        for event in events:
            # NOTE:周辺イベントの排除(ウォーミングアップやタイム)
            if event["type"] == "action":
                continue
            d = calc_time_diff(event["startTime"] ,event["endTime"])
            sec += d
            durations.append(d)
            if d <= 0:
                continue
            # print("Duration (seconds):", calc_time_diff(event["startTime"] ,event["endTime"]))
            # try:
            #     print(event["details"]["description"])
                
            # # NOTE:keyがない:KeyError,eventが辞書でない等:TypeError
            # except (KeyError, TypeError) as e:
            #     logging.warning("description 取得失敗 (%s): %s", e, event)

bins = [d/sec for d in durations]
pos = []
cur = 0
# TODO:要素数はbinの個数に一致させる　
# 評価値を正規化して(若しくは正規化された評価値を使用して)ヒートマップに反映する
# 評価値を取得できるようになったら書き直すこと
# scores = [i for i in range(len(bins))]
scores = [idx for idx, _ in enumerate(bins)]

for bin_p in bins:
    pos.append((cur,bin_p))
    cur += bin_p

cmap = plt.cm.get_cmap("viridis")
normed = [(s - min(scores)) / (max(scores) - min(scores) + 1e-6) for s in scores]
colors = [cmap(n) for n in normed]

plt.figure(figsize=(10, 2))
for (start, width), color in zip(pos, colors):
    plt.barh(0, width, left=start, color=color, edgecolor='black')

plt.xlabel("Proportional Timeline (Top Innings Only)")
plt.yticks([])
plt.title("Top Inning Events Colored by Duration Proportion")
plt.xlim(0, 1)
plt.margins(x=0)
plt.tight_layout()
plt.show()
