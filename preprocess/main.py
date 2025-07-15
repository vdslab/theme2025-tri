import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import json

from data_processor import data_process
from data_processor_for_cr import data_process_for_cr

def get_date_list():
    # start_date = datetime(2025, 3, 16)
    s_y, s_m, s_d = 2025, 3, 16
    e_y, e_m, e_d = 2025, 6, 16
    
    # start_date = datetime(2025, 3, 16)
    start_date = datetime(s_y, s_m, s_d)
    # end_date = datetime.today()
    end_date = datetime(e_y, e_m, e_d)
    
    days = (end_date - start_date).days + 1 
    return [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range(days)],s_y,s_m,s_d,e_y,e_m,e_d

def fetch_gamepks(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = requests.get(url)
    if not resp.ok:
        return []
    try:
        games = resp.json()["dates"][0]["games"]
        return [g["gamePk"] for g in games if g.get("seriesDescription") == "Regular Season" and g["status"].get("statusCode") == "F"]
    except Exception:
        return []

def output_data(process_datas_dor_rc,s_y,s_m,s_d,e_y,e_m,e_d):
    start_date_str = f"{s_y}-{s_m:02d}-{s_d:02d}"
    end_date_str = f"{e_y}-{e_m:02d}-{e_d:02d}"
    output_path = f"frontend/public/data/{start_date_str}-{end_date_str}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(process_datas_dor_rc, f, ensure_ascii=False, indent=4)
        
def main():
    process_datas_dor_rc = []
    date_str,s_y,s_m,s_d,e_y,e_m,e_d = get_date_list()
    for date in date_str:
        gamepks = fetch_gamepks(date)
        print(gamepks)
        for gamepk in gamepks:
            print(gamepk)
            raw_data,process_data = data_process(gamepk)

            process_data_dor_rc = data_process_for_cr(raw_data,process_data,gamepk)
            # print(process_data_dor_rc)
            process_datas_dor_rc.append(process_data_dor_rc)
    
    output_data(process_datas_dor_rc,s_y,s_m,s_d,e_y,e_m,e_d)
    
if __name__ == "__main__":
    main()
