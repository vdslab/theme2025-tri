import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import json
import sys
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.gamepk_to_prob import get_data
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL database connection details
DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_NAME = "public.game_detail"

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
        
def insert_records(gamepk):
    """
    gamepkを元にanalysis_dataを取得し、PostgreSQLに保存する
    """
    try:
        # analysis_dataを取得
        analysis_data = get_data(gamepk)
        
        # PostgreSQLに接続
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # UPSERT処理（gamepkが存在すれば更新、なければ挿入）
        upsert_query = f"""
            INSERT INTO {TABLE_NAME} (gamepk, analysis_data)
            VALUES (%s, %s)
            ON CONFLICT (gamepk)
            DO UPDATE SET 
                analysis_data = EXCLUDED.analysis_data
        """
        
        # analysis_dataをJSON形式で保存
        cur.execute(upsert_query, (gamepk, Json(analysis_data)))
        
        # コミット
        conn.commit()
        
        print(f"✓ gamepk {gamepk}: データを正常に更新しました")
        
    except Exception as e:
        print(f"✗ gamepk {gamepk}: エラーが発生しました - {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    
    finally:
        # 接続をクローズ
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
    
def main():
    date_str,s_y,s_m,s_d,e_y,e_m,e_d = get_date_list()
    for date in date_str:
        gamepks = fetch_gamepks(date)
        print(gamepks)
        for gamepk in gamepks:
            print(gamepk)
            insert_records(gamepk)
    
if __name__ == "__main__":
    main()
