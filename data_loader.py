import os
import random

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from calc_mean_var import load_csv
from utils import get_csv_list
import pdb

class KploacsDataset(Dataset):
    def __init__(self, base_path, target_name, stats, sample_length, stride, train=True):
        self.sample_length = sample_length

        # csv files
        if '.csv' in target_name:  # mode
            self.filelist = [target_name]
        else:  # scenario
            self.filelist = [os.path.join(base_path, target_name + '_all.csv')]

        # mean, std
        self.means, self.stds, self.maxs, self.mins = stats
        self.means[0], self.stds[0] = 0, 1  # do not normalize timestamps
        self.mins[0], self.maxs[0] = 0, 1
        self.stds[self.stds < 1e-10] = 1e-10  # prevent dividing by zero

        # handling the special case of max == min; it causes dividing by zero
        idxs = np.where(abs(self.maxs - self.mins) < 1e-10)[0]
        self.maxs[idxs] += 10
        self.mins[idxs] -= 10
        idxs_wrong = np.where(abs(self.maxs - self.mins) < 1e-10)[0]
        if len(idxs_wrong) > 0:
            print('[WARNING] wrong value range in these columns:', idxs_wrong)

        # exclude some columns in training
        dimensions_to_use = []
        for i in range(len(self.means)):
            dimensions_to_use.append(i)
        dimensions_to_use = sorted(np.array(dimensions_to_use))

        if dimensions_to_use[0] != 0:  # timestamps
            dimensions_to_use.insert(0, 0)
        self.dimensions_to_use = dimensions_to_use

        # load data
        self.data = []


        for filepath in self.filelist:
            if os.path.exists(filepath):
                header, raw_data = load_csv(filepath)
                #raw_data = pd.read_csv(filepath, delimiter=',').values.astype(np.float32)
                #raw_data = np.apply_along_axis(self.standardize, 1, raw_data, self.means, self.stds, self.dimensions_to_use)
                print("process file:"+filepath)
                raw_data = np.apply_along_axis(self.normalize, 1, raw_data, self.maxs, self.mins, self.dimensions_to_use)
                self.data.append(raw_data)
                #pdb.set_trace()

        # sampling
        self.sample_index = []
        for table_idx, single_run_data in enumerate(self.data):
            samples = list(range(0, single_run_data.shape[0] - sample_length + 1, stride))
            self.sample_index.extend([(table_idx, s) for s in samples])

    #pdb.set_trace()
    def __len__(self):
        return len(self.sample_index)


    def __getitem__(self, idx):
        table_idx, sample_idx = self.sample_index[idx]

        sample = self.data[table_idx][sample_idx:sample_idx+self.sample_length]

        timestamps = sample[:, 0]
        normalized_sample = torch.from_numpy(sample[:, 1:]).float().transpose(0, 1)  # to (dim, seq)

        return timestamps, normalized_sample

    def get_sample_dim(self):
        return len(self.dimensions_to_use) - 1

    @staticmethod
    def standardize(d, means, stds, dimensions_to_use):
        v = (d - means) / stds
        return v[dimensions_to_use]

    @staticmethod
    def normalize(d, maxs, mins, dimensions_to_use):
        v = (d - mins) / (maxs - mins)
        return v[dimensions_to_use]

    @staticmethod
    def normalize_local(d, dimensions_to_use):
        stds = np.std(d, axis=0)
        stds[stds < 1e-10] = 1e-10
        v = (d - np.mean(d, axis=0)) / stds
        return v[:, dimensions_to_use]


if __name__ == '__main__':
    ##############################################
    # TEST CODE
    ##############################################
    import config

    test_mode = 0
    base_path = './test_abnormal/normalization_sample//'

    target_description = 'Laundry'
    stats = np.load(os.path.join(base_path, target_description + '.npy'))
    #pdb.set_trace()
    stats = stats[0:4]

    base_path = './test_abnormal/'

    dataset = KploacsDataset(base_path, target_description, stats, sample_length=5, stride=3)
    #pdb.set_trace()
    loader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)

    data_iter = iter(loader)
    timestamps, batch_data = next(data_iter)
    print("Wow")
    print(timestamps)
    print(batch_data)
    print(batch_data.size())
    max, max_id = torch.max(batch_data, 1)
    min, min_id = torch.min(batch_data, 1)
    print('max: {}, index: {}'.format(max, max_id))
    print('min: {}, index: {}'.format(min, min_id))

