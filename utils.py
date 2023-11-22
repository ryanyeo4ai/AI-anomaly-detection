import numpy as np
import os
import pdb

def get_stats(cfg, name, part=0):
    stats = None
    stats = np.load(os.path.join(cfg.train_base_path, name + '.npy'))

    if part == 0:
        return stats[0:4]
    else:
        return stats[4:8]


def get_model_path(cfg, model_name, scenario_name):
    save_path = os.path.join('./model', model_name + '_' + scenario_name + '.pt')
    save_path = save_path.replace('.csv', '')
    return save_path


def get_csv_list(base_path, name=None, use_mode_group=True):
    print("get_csv_list() base_path: "+base_path)
    ret_files = []
    for root, dirs, files in os.walk(base_path):
        print("root: "+root)
        #print("dirs: "+dirs)
        #print("filies: "+files)
        for x in files:
            #print("x: "+x)
            if use_mode_group:  # mode
                #if x.startswith(name) and x.endswith('.csv') and not x.endswith('all.csv'):
                if x.endswith('.csv') and not x.endswith('all.csv'):
                    ret_files.append(os.path.join(root, x))
                    #ret_files.append(os.path.join(root+"/"+x))
                    print(ret_files)
            else:  # scenario
                if x.endswith(name + '.csv'):
                    ret_files.append(os.path.join(root, x))
        break
    return ret_files


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

