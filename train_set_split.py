import os
import csv
import shutil
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import shutil


import config as cfg
#from calc_mean_var import load_csv
from utils import get_csv_list



def splitfile():
    print(cfg.train_data_list + cfg.split_excel_name)

    load_wb = load_workbook(cfg.train_data_list + cfg.split_excel_name, data_only=True)

    load_ws = load_wb['Sheet1']

    name = cfg.mode_names[0]  # Laundry

    all_values = []

    for row in load_ws.rows:
        for cell in row:
            #print(str(cell.value))
            #print(cfg.train_data + str(cell.value) + ".csv")
            #print(cfg.train_base_path + name + "_" + str(cell.value) + ".csv")
            shutil.copy2(cfg.train_data + str(cell.value) + ".csv",
                        cfg.train_base_path + name + "_" + str(cell.value) + ".csv")
            os.remove(cfg.train_data + str(cell.value) + ".csv")

    files = os.listdir(cfg.train_data)

    for file in files:
        #print(cfg.train_data + name + "_" + file )
        shutil.copy2(cfg.train_data + file, cfg.test_data + name + "_" + file )
        os.remove(cfg.train_data + file)

  #  print(all_values)

def main(use_abnormal_file=False):
    splitfile()



if __name__ == "__main__":
    use_abnormal_file = True
    main(use_abnormal_file)
