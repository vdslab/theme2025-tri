import json
from datetime import datetime
from collections import defaultdict
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def parse_time(t):
    """ISO 8601形式の時刻文字列をdatetimeオブジェクトに変換"""
    if not t:  # 空のタイムスタンプを処理
        return None
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def merge_play_features(existing_play, new_play):
    """
    play_featuresをマージする（ORロジック）。
    1分間に三振と本塁打があれば「本塁打あり」とする。
    """
    for group_name, bool_dict in new_play.items():  # "hit_event", "rbi_impact"
        if group_name not in existing_play:
            existing_play[group_name] = bool_dict
            continue

        for feature_name, bool_val in bool_dict.items():
            if bool_val:  # 新しいイベントでTrueが立っていたら
                existing_play[group_name][
                    feature_name
                ] = True  # 既存の特徴をTrueで上書き


def time_data_sellecting(gamepk, match_data):
    """
    イベント単位のデータを「1分単位」に集約（Aggregate）する。
    """

    all_start_times = []
    all_end_times = []
    events = []

    # --- 1. 全イベントをパースし、有効な時刻を持つものだけをリスト化 ---
    for at_bat_id, event_group in match_data.items():
        for event_id, event in event_group.items():
            start_str = event.get("time", {}).get("start_time")
            end_str = event.get("time", {}).get("end_time")

            # 開始時刻または終了時刻がないイベントはスキップ
            if not start_str or not end_str:
                continue

            start = parse_time(start_str)
            end = parse_time(end_str)

            all_start_times.append(start)
            all_end_times.append(end)
            events.append(
                {
                    "start": start,
                    "end": end,
                    "play_features": event["play_features"],
                    "situation_features": event["situation_features"],
                }
            )

    if not events:
        print(f"[{gamepk}] 有効なイベントが見つかりませんでした。")
        return None

    start_time = min(all_start_times)
    end_time = max(all_end_times)
    total_duration = (end_time - start_time).total_seconds()
    total_minutes = int(total_duration // 60) + 1

    # --- 2. 1分単位に集約 (Aggregation) ---
    minute_data = {}  # defaultdict(list) から変更

    # まず、全分を空の辞書で初期化
    for m in range(total_minutes):
        minute_data[m] = None

    for event in events:
        start_minute = int((event["start"] - start_time).total_seconds() // 60)
        end_minute = int((event["end"] - start_time).total_seconds() // 60)

        event_features = {
            "play_features": event["play_features"],
            "situation_features": event["situation_features"],
        }

        for minute in range(start_minute, end_minute + 1):
            if minute not in minute_data:
                continue  # 範囲外の場合はスキップ

            if minute_data[minute] is None:
                # --- A. その分で最初のイベント ---
                # situation と play の両方を、このイベントのもので設定
                minute_data[minute] = event_features
            else:
                # --- B. その分で2番目以降のイベント ---
                # situation (状況) は、分で最初のイベントのものを維持
                # play (結果) は、マージ(OR)して「最大興奮」を反映
                existing_features = minute_data[minute]
                new_play_features = event_features["play_features"]
                merge_play_features(
                    existing_features["play_features"], new_play_features
                )

    # --- 3. 出力用に整形 ---
    output = {
        "total_duration_minutes": total_minutes,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "minutes": {},
    }

    for minute, features in minute_data.items():
        if features is None:
            # イベントがなかった分は、空の特徴量セット（あるいは前の分のコピー）を挿入
            # ここでは簡単のため、空の特徴量セットを挿入 (モデル側で window=1 が NaN で埋めてくれる)
            # ※ 本来は直前の特徴量で埋める（ffill）のが望ましい
            # print(f"[{gamepk}] minute {minute} has no events.")

            # all_plays[0] から特徴量パスを抽出するロジックのため、
            # 構造体(空のbool)は維持する必要がある。
            # ただし、モデル側の`dropna()`で結局削除される。
            # 堅牢なのは、`rf_gridsearch.py`側で`all_plays[0]`の代わりに
            # 予め定義したパスリストを使うこと。

            # → `all_plays` には None が入らないようにする
            continue

        output["minutes"][str(minute)] = [
            features
        ]  # ★ 1分に1つの集約済み特徴量をリストに入れる
        # `all_plays`のロジックを動かすため

    # --- 4. 保存 ---
    output_filename = f"data/test_molded_data/{gamepk}_test_molded_data.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"保存完了：{output_filename}")
    return output


if __name__ == "__main__":
    pk_list = [
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
    ]

    # テスト用に 1試合だけ実行
    # pk_list = ["777505"]

    for gamepk in pk_list:
        print(f"--- Processing molded data for gamepk: {gamepk} ---")
        input_filename = (
            f"data/processed_for_ra/{gamepk}_processed_for_ra_data.json"
        )

        if not os.path.exists(input_filename):
            print(f"[SKIP] File not found: {input_filename}")
            continue

        with open(input_filename, encoding="utf-8") as f:
            match_data = json.load(f)

        time_data = time_data_sellecting(gamepk, match_data)
