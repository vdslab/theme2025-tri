# gamepkを基に、ロジスティック回帰データを取得する

import json
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysys.data_collection.time_data_sellecting import time_data_sellecting
from analysys.logistic_regression_analysis.Logistic_regression_analysis import Logistic_regression_analysis
from preprocess.data_process_for_ra import data_process_for_ra
from preprocess.data_processor import data_process

def main():
    # 777398
    gamepk = input("Enter the gamepk: ")
    
    _,processed_data = data_process(gamepk)
    
    match_data = data_process_for_ra(processed_data,gamepk)
        
    molded_data = time_data_sellecting(gamepk,match_data)

    Logistic_regression_analysis(gamepk,molded_data)

def get_data(gamepk):
    
    _,processed_data = data_process(gamepk)
    
    match_data = data_process_for_ra(processed_data,gamepk)
        
    molded_data = time_data_sellecting(gamepk,match_data)

    Logistic_regression_analysis(gamepk,molded_data)

if __name__ == "__main__":
    main()