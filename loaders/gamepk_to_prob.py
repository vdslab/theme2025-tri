# gamepkを基に、分析データを取得する

import json
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.data_processor import data_process
from analysys.data_collection.data_processer import data_process_for_ra
from analysys.data_collection.time_data_sellecting import time_data_sellecting
from analysys.data_judge.catBoost_info_judge import catBoost_info_jufge

def get_data(gamepk):
    _,processed_data = data_process(gamepk)
    
    processed_data_ra = data_process_for_ra(processed_data,gamepk)
    
    molded_data = time_data_sellecting(processed_data_ra,gamepk)
    
    analysis_data = catBoost_info_jufge(molded_data)
    
    return analysis_data
    
if __name__ == "__main__":
    # 778062
    gamepk = input("Enter the gamepk: ")
    get_data(gamepk)