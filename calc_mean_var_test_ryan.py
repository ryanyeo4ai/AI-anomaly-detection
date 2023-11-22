import os
import csv
import shutil
import numpy as np
import pandas as pd

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
        dataframe = pd.read_csv(file, delimiter=',', usecols=col_selection)
    except FileNotFoundError:
        return [], []

    header = list(dataframe.columns.values)
    data = dataframe.values.astype(np.float32)
    data = data[warm_up_period:]
    #print("load_csv() : data")
    #print(data)
    return header, data


def check_validity(data):
    #time_interval = data[2, 0] - data[1, 0]
    #if time_interval >= 0.00004 or time_interval <= 0:
    #time_interval = data[2, 0] - data[1, 0] # resampling시 조절해야 함.
    #if time_interval >= 2 or time_interval <= 0.9: # resampling시 조절해야 함.
    #if time_interval >= 9 or time_interval <= 7: # resampling시 조절해야 함.
    #    return False
    #else:
    return True


''' load anomaly data for min/max computation '''
def load_auxiliary_data(filename):
    data_list = []
    for dir_name in cfg.auxiliary_path_for_stats:
        path = os.path.join(cfg.eval_base_path, dir_name, filename)
        print("load_auxiliary_data() : "+path)
        header, data = load_csv(path, cfg.warm_up_period)
        if len(data) > 0:
            data_list.append(data[:1000, :])

    return np.vstack(data_list)


def main(use_abnormal_file=False):
    path = cfg.train_base_path #eval_base_path_ryan = './test_abnormal/normalization_sample/'
    #pdb.set_trace()
    if cfg.data_selector == 'mode':
        names = cfg.mode_names
    else:
        names = cfg.scenario_names

    for data_name in names:
        print("path: " + path)
        print("data_name: "+data_name)
        files = get_csv_list(path, data_name, (cfg.data_selector == 'mode'))
        print("main() files:")
        print(files)

        data_list = []
        auxiliary_data_list = []
        header = []

        for file in files:
            print(file)
            header, data = load_csv(file, cfg.warm_up_period)
            if check_validity(data):
                data_list .append(data)
                if use_abnormal_file:
                    data_filename = os.path.basename(file)
                    print("data_filename: " + data_filename)
                    auxiliary_data = load_auxiliary_data(data_filename)
                    auxiliary_data_list.append(auxiliary_data)
            else:
                shutil.move(file, os.path.join(path, 'wrong_interval'))

        # sort data sets
        data_list.sort(key=lambda x: x[0, 0])
        auxiliary_data_list.sort(key=lambda x: x[0, 0])

        # save data
        #print("data_list:")
        #print(data_list)
        data = np.vstack(data_list)
        #print("auxiliary_data_list:")
        #print(auxiliary_data_list)
        if True:
            new_filepath = os.path.join(path, data_name+'_all.csv')
            print('writing to ' + new_filepath)
            #np.savetxt(new_filepath, data, delimiter=',', header=','.join(header), comments='') # 데이터 첫번째 줄 불균형 해결(#으로 시작하는것 수정)
            np.savetxt(new_filepath, data, delimiter=',', header=','.join(header))
            print('done')

        # normal data解決
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)

        # anomaly data
        data = np.vstack((data, np.vstack(auxiliary_data_list)))
        ext_means = np.mean(data, axis=0)
        ext_stds = np.std(data, axis=0)
        ext_maxs = np.max(data, axis=0)
        ext_mins = np.min(data, axis=0)

        # save
        np.save(os.path.join(path, data_name + '.npy'),
                [means, stds, maxs, mins, ext_means, ext_stds, ext_maxs, ext_mins])


if __name__ == "__main__":
    use_abnormal_file = True
    main(use_abnormal_file)
