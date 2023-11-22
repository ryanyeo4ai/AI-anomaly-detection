from __future__ import print_function

import glob
import os

import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt
import numpy as np

import utils
from data_loader import KploacsDataset
from vae import VAE
import config as cfg
from metric import *
import pdb

CUDA = torch.cuda.is_available()
SEED = 1

device = torch.device('cuda' if CUDA else 'cpu')
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

def evaluate_sub_for_scenario(anomaly_path, model, name):
    # define data loader
    stats = utils.get_stats(cfg, name, 1)
    dataset = KploacsDataset(anomaly_path, name, stats,
                             sample_length=cfg.sample_length, stride=cfg.sample_stride, train=False)
    if dataset.__len__() == 0:
        return  # no data file
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # load model
    model_path = utils.get_model_path(cfg, model, name)
    print("model_path: "+model_path)
    net = torch.load(model_path, map_location=torch.device(device))
    net.eval()  # set to eval mode

    print('start evaluation for ' + anomaly_path)
    timestamps_list = []
    recon_err_list = []
    feat_list = []
    for idx, (timestamps, samples) in enumerate(test_loader):
        timestamps_list.append(timestamps[:, 0])
        samples = samples.to(device)  # in shape (batch, dim, seq)

        if model == 'VAE':
            recon, regul, feat = net(samples)
        elif model == 'GANomaly':
            recon, feat, _ = net(samples)
        else:
            assert False, 'wrong model name'

        recon_diff = F.l1_loss(recon, samples, reduction='none')
        recon_diff_clamp = torch.clamp(recon_diff, min=0, max=1)
        recon_err = torch.sum(recon_diff_clamp, (1, 2))
        recon_err = recon_err.data.cpu().numpy()
        recon_err_list.append(recon_err)
        feat_list.append(feat.data.cpu().numpy())

    # aggregate
    stacked_timestamps = np.hstack(timestamps_list)
    stacked_recon_err = np.hstack(recon_err_list)

    # show
    show = False  # show or save
    anomaly_type = os.path.basename(anomaly_path)
    show_recon_error(anomaly_type, model, 'scenario_' + name, stacked_timestamps, stacked_recon_err, show)
    if show:
        plt.show()


def evaluate_sub_for_mode(anomaly_path, model, name):
    print("anomaly_path: "+anomaly_path)
    print("model: "+model)
    print("name: "+name)
    anomaly_holder = []
    # load model
    stats = utils.get_stats(cfg, name, 1)
    model_path = utils.get_model_path(cfg, model, name)
    net = torch.load(model_path, map_location=torch.device(device))
    net.eval()  # set to eval mode

    # process each csv file
    # print('start evaluation for ' + anomaly_path)
    for filepath in glob.glob(os.path.join(anomaly_path, '*.csv')):
        if (not os.path.basename(filepath).startswith(name)) or filepath.endswith('_all.csv'):
            continue

        # print(filepath)

        timestamps_list = []
        recon_err_list = []
        feat_list = []

        dataset = KploacsDataset(anomaly_path, filepath, stats,
                                 sample_length=cfg.sample_length, stride=cfg.sample_stride, train=False)

        if dataset.__len__() == 0:
            return  # no data file
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        # evaluation iteration
        for idx, (timestamps, samples) in enumerate(test_loader):
            timestamps_list.append(timestamps[:, 0])
            samples = samples.to(device)  # in shape (batch, dim, seq)

            # forward computation
            if model == 'VAE':
                recon, regul, feat  = net(samples)
            elif model == 'GANomaly':
                recon, feat, _ = net(samples)
            else:
                assert False, 'wrong model name'

            recon_diff = F.l1_loss(recon, samples, reduction='none')
            recon_diff_clamp = torch.clamp(recon_diff, min=0, max=1)
            recon_err = torch.mean(recon_diff_clamp, (1, 2))
            recon_err = recon_err.data.cpu().numpy()  # gpu -> cpu
            recon_err_list.append(recon_err)
            feat_list.append(feat.data.cpu().numpy())

        # aggregate
        stacked_timestamps = np.hstack(timestamps_list)
        stacked_recon_err = np.hstack(recon_err_list)
        stacked_feat = np.vstack(feat_list)

        anomaly_holder.append(np.max(stacked_recon_err))

        # show
        show = False  # show or save
        anomaly_type = os.path.basename(anomaly_path)
        title = os.path.basename(filepath)[:-4] + '_' + anomaly_type
        show_recon_error(title, model, 'mode_' + name, stacked_timestamps, stacked_recon_err, show)
        if show:
            plt.show()
    return anomaly_holder

def evaluate_sub_for_mode_normal(normal_path, model, name, n_total_evaluation):
    normal_holder = []
    ## calcuate reconstruction error of normal data for AUROC
    stats = utils.get_stats(cfg, name, 1)
    # load training dataset
    normal_dataset = KploacsDataset(normal_path, name, stats,
                                    sample_length=cfg.sample_length, stride=cfg.sample_stride)

    model_path = utils.get_model_path(cfg, model, name)
    net = torch.load(model_path, map_location=torch.device(device)) # load pre-trained model
    net.eval()  # set to eval mode

    # define 40% of training dataset as test normal dataset.
    # 만약 40%말고 다른걸로 하고 싶다면, 아래의 split에서 0.6을 조정하고, 또한 train_VAE.py에서도 같은 부분을 수정해야함.
    indices = list(range(int(np.shape(normal_dataset)[0])))
    split = int(np.floor(0.7 * int(np.shape(normal_dataset)[0])))
    train_indices, test_indices = indices[:split], indices[split:]
    test_sampler = SubsetRandomSampler(test_indices)
    normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                drop_last=False,
                                                sampler=test_sampler)
    # define reconstruction error holder
    recon_err_list_normal = []

    for idx_normal, (timestamps_normal, samples_normal) in enumerate(normal_loader):
        samples_normal = samples_normal.to(device)  # in shape (batch, dim, seq)

        # forward computation
        if model == 'VAE':
            recon_normal, regul_normal, feat_normal = net(samples_normal)
        elif model == 'GANomaly':
            recon_normal, feat_normal, _ = net(samples_normal)
        else:
            assert False, 'wrong model name'

        recon_diff_normal = F.l1_loss(recon_normal, samples_normal, reduction='none')
        recon_diff_clamp_normal = torch.clamp(recon_diff_normal, min=0, max=1)
        recon_err_normal = recon_diff_clamp_normal.data.cpu().numpy() # gpu -> cpu
        if idx_normal == 0:
            recon_err_list_normal = recon_err_normal
        else:
            recon_err_list_normal = np.concatenate((recon_err_list_normal, recon_err_normal), axis = 0)

    mean_recon_err_normal = np.mean(recon_err_list_normal, axis = (1,2))
    n_data_per_mode_scenario = int(np.shape(normal_dataset)[0]//n_total_evaluation) # (시나리오-모드) 당 샘플수
    # calculate anomaly score of each scenario using mean operation
    n_iter_score = (len(mean_recon_err_normal) // n_data_per_mode_scenario) + 1
    for n in range(n_iter_score):
        # 아노말리 스코어는 각 모드 및 시나리오 마다 구해줌.
        if n != (n_iter_score-1):
            normal_holder.append(np.max(mean_recon_err_normal[n*n_data_per_mode_scenario: (n+1)*n_data_per_mode_scenario]))
        else:
            normal_holder.append(np.max(mean_recon_err_normal[n*n_data_per_mode_scenario: ]))
    return normal_holder


def show_recon_error(title, model, name, timestamps, errors, show):
    plt.plot(timestamps, errors)
    plt.title(title)
    plt.xlabel('time(s)', fontsize=16)
    plt.ylabel('reconstruction error', fontsize=16)
    plt.xticks(size=14)
    plt.yticks(size=14)

    if not show:  # save
        out_path = os.path.join(cfg.out_path, model, name)
        utils.make_directory(out_path)
        np.savetxt(os.path.join(out_path, title + '.csv'), np.vstack((timestamps, errors)))  # csv file save
        plt.savefig(os.path.join(out_path, title + '.png'))
        plt.clf()


def evaluate(model, name):
    anomaly_score = []
    # calcuate anomaly score of abnormal data
    subfolders = [f.path for f in os.scandir(cfg.eval_base_path) if f.is_dir()]
    n_total_evaluation = 10
    for i, subfolder in enumerate(subfolders):
        if cfg.data_selector == 'mode':
            anomaly_temp = evaluate_sub_for_mode(subfolder, model, name)
            n_total_evaluation += len(anomaly_temp)
            for j in range(len(anomaly_temp)):
                anomaly_score.append(anomaly_temp[j])
        else:
            evaluate_sub_for_scenario(subfolder, model, name)

    # calculate anomaly score of normal data(40% of training normal dataset)
    anomaly_score = np.reshape(anomaly_score, [-1])
    normal_score = evaluate_sub_for_mode_normal(cfg.train_base_path, model, name, n_total_evaluation)
    # make normal close to 0 and abnormal close to 1
    normal_score = np.subtract([1]*len(normal_score) ,normal_score)
    anomaly_score = np.subtract([1]*len(anomaly_score), anomaly_score)
    # calcuate various metric for evaluating binary classification
    metrics = metric(normal_score, anomaly_score)

    return metrics['AUROC']


if __name__ == '__main__':

    # select model
    model = 'GANomaly'  # {VAE, GANomaly}

    if cfg.data_selector == 'mode':
        eval_names = cfg.mode_names
    else:
        eval_names = cfg.scenario_names

    for name in eval_names:
        print(name)
        AUROC = evaluate(model, name)
        print(AUROC)

