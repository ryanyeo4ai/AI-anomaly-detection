import torch
import torch.nn as nn
import config as cfg
from config import calc_dim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.latent_dim = 100
#        self.before_latent_dim = (cfg.sample_length - 6) * int(dim / 8)
        self.before_latent_dim = (cfg.sample_length - 6) * calc_dim(dim, 8) 
        self.dim = dim
        self.encoder = nn.Sequential(
#            nn.Conv1d(dim, int(dim/2), 3),
            nn.Conv1d(dim, calc_dim(dim,2), 3),
#            nn.BatchNorm1d(int(dim/2)),
            nn.BatchNorm1d(calc_dim(dim, 2)),
            nn.ReLU(True),
#            nn.Conv1d(int(dim/2), int(dim/4), 3),
            nn.Conv1d(calc_dim(dim,2), calc_dim(dim,4), 3),
#            nn.BatchNorm1d(int(dim / 4)),
            nn.BatchNorm1d(calc_dim(dim , 4)),
            nn.ReLU(True),
#            nn.Conv1d(int(dim / 4), int(dim / 8), 3),
            nn.Conv1d(calc_dim(dim , 4), calc_dim(dim , 8), 3),
#            nn.BatchNorm1d(int(dim / 8)),
            nn.BatchNorm1d(calc_dim(dim , 8)),
            nn.ReLU(True)
        )
        self.encoder_fc = nn.Linear(self.before_latent_dim, self.latent_dim)

    def forward(self, x):
        output = self.encoder(x)
        output = output.view(-1, self.before_latent_dim)
        feat = self.encoder_fc(output)
        return feat


class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.latent_dim = 100
#        self.before_latent_dim = (cfg.sample_length - 6) * int(dim / 8)
        self.before_latent_dim = (cfg.sample_length - 6) * calc_dim(dim , 8)
        self.dim = dim

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.before_latent_dim),
            nn.BatchNorm1d(self.before_latent_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
#            nn.ConvTranspose1d(int(dim / 8), int(dim / 4), 3, stride=1),
            nn.ConvTranspose1d(calc_dim(dim , 8), calc_dim(dim , 4), 3, stride=1),
#            nn.BatchNorm1d(int(dim / 4)),
            nn.BatchNorm1d(calc_dim(dim , 4)),
            nn.ReLU(True),
#            nn.ConvTranspose1d(int(dim / 4), int(dim / 2), 2, stride=1),
            nn.ConvTranspose1d(calc_dim(dim , 4), calc_dim(dim , 2), 2, stride=1),
#            nn.BatchNorm1d(int(dim / 2)),
            nn.BatchNorm1d(calc_dim(dim , 2)),
            nn.ReLU(True),
#            nn.ConvTranspose1d(int(dim / 2), dim, 4, stride=1)
            nn.ConvTranspose1d(calc_dim(dim , 2), dim, 4, stride=1)
        )

    def forward(self, x):
        output = self.decoder_fc(x)
#        output = output.view(-1, int(self.dim / 8), (cfg.sample_length - 6))
        output = output.view(-1, calc_dim(self.dim , 8), (cfg.sample_length - 6))
        output = self.decoder(output)
        return output


class NetD(nn.Module):
    def __init__(self, data_dim):
        super(NetD, self).__init__()
        self.encoder = Encoder(data_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        classifier = self.classifier(features)
        classifier = classifier.squeeze(1)

        return classifier, features


class NetG(nn.Module):
    def __init__(self, data_dim):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(data_dim)
        self.decoder = Decoder(data_dim)
        self.encoder2 = Encoder(data_dim)

    def forward(self, x):
        latent_i = self.encoder1(x)
        recon = self.decoder(latent_i)
        latent_o = self.encoder2(recon)
        return recon, latent_i, latent_o
