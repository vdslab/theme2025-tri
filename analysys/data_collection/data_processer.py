import json
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_scores(processed_data):
    
    game_data = processed_data
    scores_data = {}
    for p_idx, play in game_data.items():
        p_score = {}
        for e_idx, event in play.items():
            # アクション以外のイベント（ウォームアップ等）はスキップ
            if (
                event["event_type"] == "action"
                and event["is_base_running_play"] == None
            ):
                continue
            e_score = {}
            play_features = {}
            situation_features = {}

            # 特徴量エンジニアリング関数を呼び出す
            get_play_score(event, play_features)
            get_situation_score(event, situation_features) # ★★★ ここが更新対象 ★★★

            p_score[e_idx] = e_score
            e_score["play_features"] = play_features
            e_score["situation_features"] = situation_features
            e_score["time"] = event["time"]
            e_score["detail"] = event["detail"]

        scores_data[p_idx] = p_score
    return scores_data


def get_play_score(event, play_features):
    """
    プレイの「結果」に関する特徴量 (play_features) を生成する。
    """
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
                pre_score_diff = (
                    team_score["home"]["pre_score"] - team_score["away"]["pre_score"]
                )
                if pre_score_diff < rbi:  #pre_score_diff < rbi で逆転
                    if event["is_last_inning"]:
                         # 9回表の逆転はサヨナラではない
                         rbi_impact["go_ahead_rbi"] = True
                    else:
                         rbi_impact["go_ahead_rbi"] = True
                elif pre_score_diff == rbi:
                    rbi_impact["tie_rbi"] = True
                else: # pre_score_diff > rbi
                    rbi_impact["regular_rbi"] = True
            else:
                rbi_impact["regular_rbi"] = True

        else: # Home (裏) の攻撃
            if team_score["away"]["pre_score"] - team_score["home"]["pre_score"] >= 0: # 負けているか同点
                pre_score_diff = (
                    team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
                )
                if pre_score_diff < rbi: # 逆転
                    if event["is_last_inning"]:
                        rbi_impact["sayonara_rbi"] = True # 最終回裏の逆転はサヨナラ
                    else:
                        rbi_impact["go_ahead_rbi"] = True
                elif pre_score_diff == rbi:
                    rbi_impact["tie_rbi"] = True
                else: # pre_score_diff > rbi
                     rbi_impact["regular_rbi"] = True
            else:
                rbi_impact["regular_rbi"] = True

    play_features["hit_event"] = hit_event
    play_features["rbi_impact"] = rbi_impact


def get_situation_score(event, situation_features):
    """
    プレイ開始「前」の「状況」に関する特徴量 (situation_features) を生成する。
    """
    
    # --- 1. ランナー状況 (変更なし) ---
    runner_status = {}
    runner_state = event["runner_state"]
    is_runner = {"1B": False, "2B": False, "3B": False}
    # NOTE:イベント前はpre,イベント後はpos
    for k, v in runner_state["pre_runner_state"].items():
        if v["id"]:
            is_runner[k] = True

    runner_status["none"] = (
        not is_runner["1B"] and not is_runner["2B"] and not is_runner["3B"]
    )
    runner_status["first"] = (
        is_runner["1B"] and not is_runner["2B"] and not is_runner["3B"]
    )
    runner_status["second"] = (
        not is_runner["1B"] and is_runner["2B"] and not is_runner["3B"]
    )
    runner_status["third"] = (
        not is_runner["1B"] and not is_runner["2B"] and is_runner["3B"]
    )
    runner_status["first-second"] = (
        is_runner["1B"] and is_runner["2B"] and not is_runner["3B"]
    )
    runner_status["first-third"] = (
        is_runner["1B"] and not is_runner["2B"] and is_runner["3B"]
    )
    runner_status["second-third"] = (
        not is_runner["1B"] and is_runner["2B"] and is_runner["3B"]
    )
    runner_status["first-second-third"] = (
        is_runner["1B"] and is_runner["2B"] and is_runner["3B"]
    )
    
    # --- (既存の特徴量: is_goahead_runner_on_base - 変更なし) ---
    is_goahead_runner_on_base = False
    pre_runner_count = event["runner_count"]["pre_runner_count"]
    team_score = event["team_score"]
    pre_score_diff_abs = 0 # 守備側から見た点差
    if event["is_away"]:
        pre_score_diff_abs = (
            team_score["home"]["pre_score"] - team_score["away"]["pre_score"]
        )
    else:
        pre_score_diff_abs = (
            team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
        )
    
    # 負けていて、ランナーの数が点差を上回っている場合
    if pre_score_diff_abs > 0 and pre_runner_count > pre_score_diff_abs:
        is_goahead_runner_on_base = True


    # --- 2. 点差 (変更なし) ---
    score_difference = {}
    pre_score_diff_batting = 0 # 攻撃側から見た点差
    if event["is_away"]:
        pre_score_diff_batting = (
            team_score["away"]["pre_score"] - team_score["home"]["pre_score"]
        )
    else:
        pre_score_diff_batting = (
            team_score["home"]["pre_score"] - team_score["away"]["pre_score"]
        )
        
    score_difference["minus_less_3"] = pre_score_diff_batting <= -3
    score_difference["minus_2"] = pre_score_diff_batting == -2
    score_difference["minus_1"] = pre_score_diff_batting == -1
    score_difference["tie"] = pre_score_diff_batting == 0
    score_difference["plus_1"] = pre_score_diff_batting == 1
    score_difference["plus_2"] = pre_score_diff_batting == 2
    score_difference["plus_more_3"] = pre_score_diff_batting >= 3

    
    # --- 3. イニング (バグ修正) ---
    inning_phase = {}
    inning = event["inning"]
    inning_phase["early"] = inning <= 3
    inning_phase["middle"] = inning >= 4 and inning <= 6 # バグ修正
    inning_phase["late"] = inning >= 7                   # バグ修正

    # --- 4. イニングの文脈 (新規追加) ---
    inning_context = {}
    inning_context["is_extra_inning"] = inning >= 10
    # is_last_inning は前段の data_process.py で計算済みのブール値
    inning_context["is_last_inning"] = event.get("is_last_inning", False)
    
    # --- 5. アウトカウント (新規追加) ---
    outs_status = {}
    # event["detail"]["count"] からアウトカウントを取得 (存在しない場合 0)
    outs = event["detail"].get("count", {}).get("outs", 0) 
    outs_status["outs_0"] = outs == 0
    outs_status["outs_1"] = outs == 1
    outs_status["outs_2"] = outs == 2

    # --- 6. 打席カウント (新規追加) ---
    count_status = {}
    count = event["detail"].get("count", {})
    balls = count.get("balls", 0)
    strikes = count.get("strikes", 0)
    
    # --- 7. 交互作用特徴量 (新規追加) ---
    interaction_features = {}
    
    # スタッツによる特徴量(11/5 追加)
    # NOTE: シーズン打率、ホームラン数、 ops、 本試合のヒット回数 を取得
    # NOTE: 閾値は仮決め
    stats = event["stats"]
    
    season_avg = {}
    season_avg["high"] = float(stats["season_avg"]) >= 0.30
    season_avg["middle"] = 0.30 > float(stats["season_avg"]) >= 0.28
    season_avg["low"] = 0.28 > float(stats["season_avg"])
    season_home_runs = {}
    season_home_runs["high"] = float(stats["season_home_runs"]) >= 25
    season_home_runs["middle"] = 25 > float(stats["season_home_runs"]) >= 10
    season_home_runs["low"] = 10 > float(stats["season_home_runs"])
    # season_ops = stats["season_ops"] > 0.8
    # today_hits = stats["today_hits"]

    # ローカル変数から値を取得（まだsituation_featuresには代入されていないため）
    is_late = inning_phase["late"]
    is_tie = score_difference["tie"]
    is_minus_1 = score_difference["minus_1"]
    is_2_outs = outs_status["outs_2"]
    is_scoring_pos = (
        runner_status["second"] or
        runner_status["third"] or
        runner_status["first-second"] or
        runner_status["first-third"] or
        runner_status["second-third"] or
        runner_status["first-second-third"]
    )

    # 「終盤」かつ「僅差」か
    interaction_features["is_late_and_close"] = is_late and (is_tie or is_minus_1)

    # 「終盤」かつ「2アウト」かつ「得点圏」か (最高レベルの緊迫度)
    interaction_features["is_high_leverage"] = is_late and is_2_outs and is_scoring_pos
    

    count_status["is_full_count"] = balls == 3 and strikes == 2
    # (オプション: 興奮に繋がりやすいため追加)
    count_status["is_advantage_batter"] = balls == 3 and strikes < 2
    count_status["is_advantage_pitcher"] = strikes == 2 and balls < 3

    # --- 最終的な特徴量を辞書に格納 ---
    situation_features["runner_status"] = runner_status
    situation_features["is_goahead_runner_on_base"] = is_goahead_runner_on_base
    situation_features["score_difference"] = score_difference
    situation_features["inning_phase"] = inning_phase       # 修正済み
    situation_features["inning_context"] = inning_context   # 新規
    situation_features["outs_status"] = outs_status         # 新規
    situation_features["count_status"] = count_status       # 新規
    situation_features["interaction_features"] = interaction_features # 新規
    situation_features["season_avg"] = season_avg
    situation_features["season_home_runs"] = season_home_runs


def output_data(scores_data, gamepk):

    output_path = f"data/test_processed_for_ra/{gamepk}_processed_for_ra_data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, ensure_ascii=False, indent=4)


def data_process_for_ra(processed_data, gamepk):

    scores_data = get_scores(processed_data)
    output_data(scores_data, gamepk)

    return scores_data


if __name__ == "__main__":
    # (変更なし)
    for gamepk in [
        "778199",
        "777579",
        "777863",
        "777940",
        "777571",
        "777988",
        "778062",
        "778434",
        "777701",
        "778444",
        "777649",
        "778220",
        "778406",
        "778544",
        "777726",
        "778285",
        "778262",
        "778163",
        "777505",
    ]:
        input_path = f"data/processed/{gamepk}_processed_data.json"
        print(f"Processing gamepk: {gamepk}")
        with open(input_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        data_process_for_ra(processed_data, gamepk)
