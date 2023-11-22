import os
import csv
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from utils import get_csv_list
import pdb


def load_csv(file, warm_up_period=cfg.warm_up_period):

    def col_selection(c):
        if c not in cfg.exclude_columns:
            return True
        else:
            return False

    try:

        #dataframe = pd.read_csv(file, delimiter=',', header=None, usecols=col_selection)
        dataframe = pd.read_csv(file, delimiter=',', header=0, usecols=['no','PM25'])
        #sc = MinMaxScaler()
        #dataframe = sc.fit_transform(dataframe)
        #dataframe=pd.DataFrame(dataframe)
        ''' 
        #dataframe = dataframe.transpose()

        target_data = dataframe[0][0]

        #dataframe = dataframe[1:].reset_index(drop=True)
        #dataframe = dataframe[cfg.col_names].reset_index(drop=True)
        addlist = [0]
        cnt = 0
        
        for i in dataframe.index:
            tmp = (1/target_data * cnt) + 1/target_data
            addlist.append(round(tmp, 10))
            cnt = cnt + 1
        

        dataframe[1] = pd.Series(addlist)
        dataframe = pd.DataFrame(dataframe, columns=[1, 0])
        print(dataframe)
        '''
        header = list([0,1])
        data = dataframe.values.astype(np.float32)
        #data = dataframe.loc[:,[1,2]]
    except FileNotFoundError:
        return [], []

    #header = list(dataframe.columns.values)
    

    return header, data

#eval_base_path_ryan = './test_abnormal/normalization_sample/'
#train_base_path = './train_normal/'
#test_base_path = './test_normal/'
def main(use_abnormal_file=False):
    #path = cfg.train_base_path 
    path = cfg.train_data
    #path = 'train_normal'
    if cfg.data_selector == 'mode':
        names = cfg.mode_names #mode_names = ['PM25']
    else:
        names = cfg.scenario_names
    print(names)
    for data_name in names:
        print(data_name)
        print(path)
        files = get_csv_list(path, data_name, (cfg.data_selector == 'mode'))
        print(files)
        print('# of files : {}'.format(len(files)))
        for file in files:
            print("main() file: "+file)
            header, data = load_csv(file, cfg.warm_up_period) #warm_up_period = 1
            base_file_name = os.path.basename(file)
            #new_filepath = os.path.splitext(file)[0] + "_convert.csv"
            new_filepath = os.path.join(cfg.train_base_path, os.path.splitext(base_file_name)[0] + "_convert.csv")

            print(new_filepath)
            pd.DataFrame(data).to_csv(new_filepath, index=None)
            #np.savetxt(new_filepath, data, fmt="%10f", delimiter=',', header=','.join(header))
            os.remove(os.path.splitext(file)[0] + ".csv")
    '''
    # for test_abnormal/normalization_sample/   
    path_test = cfg.test_base_path
    for data_name in names:
        files = get_csv_list(path_test, data_name, (cfg.data_selector == 'mode'))
        print('# of files : {}'.format(len(files)))
        for file in files:
            print(file)
            header, data = load_csv(file, cfg.warm_up_period) #warm_up_period = 1

            new_filepath = os.path.splitext(file)[0] + "_convert.csv"

            print(new_filepath)
            pd.DataFrame(data).to_csv(new_filepath, index=None)
            #np.savetxt(new_filepath, data, fmt="%10f", delimiter=',', header=','.join(header))
            #os.remove(os.path.splitext(file)[0] + ".csv")
    '''        
if __name__ == "__main__":
    use_abnormal_file = True
    main(use_abnormal_file)
