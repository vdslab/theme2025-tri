import requests
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import os

# 日付を設定（例: 2025年6月10日）
target_date = "2025-06-10"
sport_id = 1  # MLB

def get_schedule(date):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId={sport_id}&date={date}"
    return requests.get(url).json()

def get_game_pbp(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    return requests.get(url).json()

def extract_features(play, inning):
    features = {}

    # ヒット種別
    result = play.get("result", {})
    event = result.get("eventType", "").lower()
    features.update({
        "single": event == "single",
        "double": event == "double",
        "triple": event == "triple",
        "home_run": event == "home_run",
        "hit_other": event in ["walk", "field_error", "hit_by_pitch"]
    })

    # 打点と点差影響
    rbi = result.get("rbi", 0)
    is_walkoff = "walkOff" in result.get("description", "").lower()

    # スコア（安全に取得）
    home_score = result.get("homeScore", 0)
    away_score = result.get("awayScore", 0)
    before_score = home_score - away_score
    after_score = before_score + rbi

    is_tie = before_score != 0 and after_score == 0
    is_lead = (before_score < 0 and after_score > 0) or (before_score > 0 and after_score < 0)

    features.update({
        "rbi_normal": rbi > 0 and not (is_tie or is_lead or is_walkoff),
        "rbi_tie": is_tie,
        "rbi_lead": is_lead,
        "rbi_walkoff": is_walkoff
    })

    # ランナー状況
    runners = play.get("matchup", {}).get("baseRunners", [])
    runner_on = [r.get("base", "") for r in runners]
    bases = {
        "1B": "runner_1b_only",
        "2B": "runner_2b_or_3b",
        "3B": "runner_2b_or_3b",
        "FULL": "bases_loaded",
        "NONE": "runner_none"
    }
    base_flags = {
        "runner_1b_only": False,
        "runner_2b_or_3b": False,
        "runner_23": False,
        "bases_loaded": False,
        "runner_none": False
    }

    bases_set = set(runner_on)
    if bases_set == {"1B"}:
        base_flags["runner_1b_only"] = True
    elif bases_set == {"2B"} or bases_set == {"3B"}:
        base_flags["runner_2b_or_3b"] = True
    elif "2B" in bases_set and "3B" in bases_set:
        base_flags["runner_23"] = True
    elif all(b in bases_set for b in ["1B", "2B", "3B"]):
        base_flags["bases_loaded"] = True
    elif not bases_set:
        base_flags["runner_none"] = True

    features.update(base_flags)

    # 逆転ランナー
    features["has_go_ahead_runner"] = "goAhead" in result.get("description", "").lower()

    # 相対得点差
    score_diff = abs(before_score)
    features.update({
        "score_tie": before_score == 0,
        "score_diff_1": score_diff == 1,
        "score_diff_2": score_diff == 2,
        "score_diff_3_or_more": score_diff >= 3
    })

    # 試合の局面
    if 1 <= inning <= 3:
        features["inning_early"] = True
        features["inning_mid"] = False
        features["inning_late"] = False
    elif 4 <= inning <= 6:
        features["inning_early"] = False
        features["inning_mid"] = True
        features["inning_late"] = False
    else:
        features["inning_early"] = False
        features["inning_mid"] = False
        features["inning_late"] = True

    return features

def main():
    schedule = get_schedule(target_date)
    games = schedule["dates"][0]["games"]

    output = {}

    for game in tqdm(games, desc="Processing games"):
        game_pk = game["gamePk"]
        game_data = get_game_pbp(game_pk)

        plays = []
        for inning in game_data.get("liveData", {}).get("plays", {}).get("allPlays", []):
            start_time_str = inning["about"].get("startTime")
            if not start_time_str:
                continue

            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            minute_bucket = start_time.replace(second=0, microsecond=0)
            features = extract_features(inning, inning["about"]["inning"])

            plays.append({
                "timestamp": minute_bucket.isoformat(),
                "features": features
            })

        output[game_pk] = plays

    os.makedirs("output", exist_ok=True)
    with open(f"output/mlb_features_{target_date}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"保存しました: output/mlb_features_{target_date}.json")

if __name__ == "__main__":
    main()
