from __future__ import print_function

import os

import torch
import torch.utils.data
from torch import nn, optim
from matplotlib import pyplot as plt
import numpy as np

import utils
from data_loader import KploacsDataset
import ganomaly_net
import config as cfg


CUDA = torch.cuda.is_available()
SEED = 1

device = torch.device('cuda' if CUDA else 'cpu')
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)


def train(name):
    # define data loader
    stats = utils.get_stats(cfg, name, 1)
    dataset = KploacsDataset(cfg.train_base_path, name, stats,
                             sample_length=cfg.sample_length, stride=cfg.sample_stride)
    sample_dim = dataset.get_sample_dim()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # define model
    gen = ganomaly_net.NetG(sample_dim).to(device)
    dis = ganomaly_net.NetD(sample_dim).to(device)
    optimizer_gen = optim.Adam(gen.parameters(), lr=cfg.lr)
    optimizer_dis = optim.Adam(dis.parameters(), lr=cfg.lr)
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # init variables
    real_label = torch.ones(size=(cfg.batch_size,), dtype=torch.float32).to(device)
    fake_label = torch.zeros(size=(cfg.batch_size,), dtype=torch.float32).to(device)

    # let's train
    print('start training for ' + name)
    gen.train()  # set to train mode
    dis.train()
    loss_history = []
    for epoch in range(cfg.epochs):
        running_loss = np.array([0.0, 0.0, 0.0, 0.0])
        for idx, (timestamps, samples) in enumerate(train_loader):
            samples = samples.to(device)  # in shape (batch, dim, seq)

            ####################################################
            # update discriminator
            optimizer_dis.zero_grad()

            # real samples
            out_d_real, feat_real = dis(samples)

            # fake samples
            fake, latent_i, latent_o = gen(samples)
            out_d_fake, feat_fake = dis(fake.detach())

            err_d = l2_loss(feat_real, feat_fake)
            err_d.backward()
            optimizer_dis.step()

            ####################################################
            # update generator
            optimizer_gen.zero_grad()
            out_g, _ = dis(fake)

            err_g_bce = bce_loss(out_g, real_label)
            err_g_l1l = l1_loss(fake, samples)
            err_g_enc = l2_loss(latent_o, latent_i)
            err_g = err_g_bce * 1 + err_g_l1l * 50 + err_g_enc * 1

            err_g.backward(retain_graph=True)
            optimizer_gen.step()

            # print
            running_loss += np.array([err_d.item(), err_g_bce.item(), err_g_l1l.item(), err_g_enc.item()])
            if idx % cfg.log_interval == cfg.log_interval - 1:
                loss_val = running_loss / cfg.log_interval
                running_loss.fill(0.0)
                loss_val_str = ', '.join('{:.3f}'.format(v) for v in loss_val)
                print_str = "[{}/{}, {:5d}] loss: {}".format(epoch + 1, cfg.epochs, idx + 1, loss_val_str)
                loss_history.append(loss_val)
        print(print_str)

    # save model
    #torch.save(gen, utils.get_model_path(cfg, 'GANomaly', name))
    torch.save(gen, './model/GANomaly_PM25.pt')

    # show loss history
    #loss_history = np.array(loss_history)
    #for i in range(loss_history.shape[1]):
    #    plt.plot(loss_history[:, i])
    #    plt.show()


if __name__ == '__main__':
    if cfg.data_selector == 'mode':
        names = cfg.mode_names
    else:
        names = cfg.scenario_names
    print(names)

    for name in names:
        train(name)
