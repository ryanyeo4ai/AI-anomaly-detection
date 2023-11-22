from __future__ import print_function

import os
import glob
from matplotlib import pyplot as plt
import numpy as np
from Bayesian_op.bayes_opt import BayesianOptimization
import random
import pdb

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import utils
from data_loader import KploacsDataset
from vae import VAE
import config as cfg
from eval import evaluate

CUDA = torch.cuda.is_available()
SEED = 1

device = torch.device('cuda' if CUDA else 'cpu')
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

def train(init_learning_rate_log, lamda):
    # define data loader
    stats = utils.get_stats(cfg, name, 1)

    dataset = KploacsDataset(cfg.train_base_path, name, stats,
                             sample_length=cfg.sample_length, stride=cfg.sample_stride)


    # split training data into two group: 60% training dataset, 40% test dataset
    indices = list(range(int(np.shape(dataset)[0])))
    split = int(np.floor(0.7*int(np.shape(dataset)[0])))
    train_indices, test_indices = indices[:split], indices[split:]
    random.shuffle(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    sample_dim = dataset.get_sample_dim()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                               shuffle=False, drop_last=True, sampler = train_sampler)

    # Define hyperparameter for bayesian optimization(바꾼 부분)
    hp_d['learning_rate'] = 10**init_learning_rate_log
    hp_d['lamda'] = lamda

    # define model
    net = VAE(sample_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=hp_d['learning_rate'])
    criterion = nn.MSELoss()

    # let's train

    # print('start training for ' + name)
    net.train()  # set to train mode
    loss_history = []
    for epoch in range(cfg.epochs):
        running_loss = 0.00
        for idx, (timestamps, samples) in enumerate(train_loader):
            samples = samples.to(device)  # in shape (batch, dim, seq)

            optimizer.zero_grad()

            # forward
            recon, regul, feat = net(samples)
            loss = criterion(recon, samples) + hp_d['lamda']*regul

            # backward
            loss.backward()
            optimizer.step()

            # print
            running_loss += loss.item()
            if idx % cfg.log_interval == cfg.log_interval - 1:
                loss_val = running_loss / cfg.log_interval
                running_loss = 0
                print_str = "[{}/{}, {:5d}] total_loss: {:.3f}".format(epoch + 1, cfg.epochs, idx + 1, loss_val)
                loss_history.append(loss_val)
                # if cfg.bayesian is False:
                # print(print_str)

    # save model
    #torch.save(net, utils.get_model_path(cfg, 'VAE', name))
    torch.save(net, './model/VAE_PM25.pt')
    # evaluate model with AUROC for applying bayesian optimization(바꾼 부분)
    model = 'VAE'
    AUROC = evaluate(model, name=name)
    print("AUROC")
    print(AUROC)
    return AUROC


if __name__ == '__main__':
    # cfg.bayesian: if True than applying bayesian optimization
    # else just training neural networks via hyperparameters in cfg
    hp_d = dict()
    hp_d['lr'] = -2
    hp_d['lamda'] = 0.8


    if cfg.data_selector == 'mode':
        names = cfg.mode_names
    else:
        names = cfg.scenario_names
    print(names)

    for name in names:  # train each mode or scenario
        # Define BayesianOptimization object
        print('start training for ' + name)
        bayes_optimizer = BayesianOptimization(
            f=train,
            pbounds={
                'init_learning_rate_log': (-5, -3),
                'lamda': (0.2, 0.8)
                },
            random_state=0,
            verbose=2
        )
        #pdb.set_trace()
        # applying bayesian optimization
        bayes_optimizer.maximize(init_points=3, n_iter=10, acq='ei', xi=0.01)  # FIXME

        for i, res in enumerate(bayes_optimizer.res):
            print('Iteration {}: \n\t{}'.format(i, res))
        print('Final result: ', bayes_optimizer.max) # 바꾼 부분 끝
        print(bayes_optimizer.res)
