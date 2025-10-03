import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import json
import psycopg2
import psycopg2.extras

from data_processor import data_process
from data_processor_for_cr import data_process_for_cr

def get_date_list():
    # start_date = datetime(2025, 3, 16)
    s_y, s_m, s_d = 2025, 3, 16
    e_y, e_m, e_d = 2025, 7, 28
    
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
            raw_data,meta,process_data = data_process(gamepk)

            process_data_dor_rc = data_process_for_cr(raw_data,meta,process_data,gamepk)
            # print(process_data_dor_rc)
            process_datas_dor_rc.append(process_data_dor_rc)
    
    output_data(process_datas_dor_rc,s_y,s_m,s_d,e_y,e_m,e_d)
    
# def main():
#     insert_data = []
#     date_str,s_y,s_m,s_d,e_y,e_m,e_d = get_date_list()
#     for date in date_str:
#         gamepks = fetch_gamepks(date)
#         print(gamepks)
#         for gamepk in gamepks:
#             print(gamepk)
#             data = {}
#             process_data = data_process(gamepk)
#             data["gamepk"] = gamepk
#             data["play_data"] = process_data
#             data["analysis_data"] = {}
#             data["movie_label"] = []
#             insert_data.append(data)
#     chunk_and_insert_all(insert_data)
            
# def chunk_and_insert_all(insert_data, chunk_size=100):
#     for i in range(0, len(insert_data), chunk_size):
#         chunk = insert_data[i:i+chunk_size]
#         insert_to_postgresql(chunk)

# # PostgreSQL database connection details
# DATABASE_URL = "postgresql://postgres.kvnjgvidsowvnnbaazyy:kohsei0720meimei@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres"
# TABLE_NAME = "game_data"

# def insert_to_postgresql(data_chunk):
#     try:
#         # Connect to PostgreSQL database
#         conn = psycopg2.connect(DATABASE_URL)
#         cursor = conn.cursor()
        
#         # Prepare the INSERT statement
#         insert_query = f"""
#         INSERT INTO {TABLE_NAME} (gamepk, play_data, analysis_data, movie_label)
#         VALUES (%s, %s, %s, %s)
#         ON CONFLICT (gamepk) DO UPDATE SET
#         play_data = EXCLUDED.play_data,
#         analysis_data = EXCLUDED.analysis_data,
#         movie_label = EXCLUDED.movie_label
#         """
        
#         # Prepare data for insertion
#         values_to_insert = []
#         for record in data_chunk:
#             values_to_insert.append((
#                 record["gamepk"],
#                 json.dumps(record["play_data"]),  # Convert to JSON string for JSONB
#                 json.dumps(record["analysis_data"]) if record["analysis_data"] else None,  # Convert to JSON string for JSONB
#                 record["movie_label"]  # Keep as array for ARRAY type
#             ))
        
#         # Execute the batch insert
#         cursor.executemany(insert_query, values_to_insert)
        
#         # Commit the transaction
#         conn.commit()
        
#         print(f"Successfully inserted {len(data_chunk)} records into {TABLE_NAME}")
        
#     except psycopg2.Error as e:
#         print(f"Database error: {e}")
#         if 'conn' in locals():
#             conn.rollback()
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()

if __name__ == "__main__":
    main()
