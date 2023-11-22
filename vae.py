import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.autograd import Variable
import config as cfg
import pdb
import numpy as np

class VAE(nn.Module):
    def __init__(self, dim):
        super(VAE, self).__init__()
        self.latent_dim = 100
#       self.before_latent_dim = (cfg.sample_length - 6) * int(dim / 8)
        self.before_latent_dim = (cfg.sample_length - 6) * dim
        self.dim = dim

        # encoder
        self.encoder = nn.Sequential(
#            nn.Conv1d(dim, int(dim / 2), 3),
            nn.Conv1d(dim, dim, 3),
#            nn.BatchNorm1d(int(dim / 2)),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
#            nn.Conv1d(int(dim / 2), int(dim / 4), 3),
            nn.Conv1d(dim, dim, 3),
#            nn.BatchNorm1d(int(dim / 4)),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
#            nn.Conv1d(int(dim / 4), int(dim / 8), 3),
            nn.Conv1d(dim, dim, 3),
#            nn.BatchNorm1d(int(dim / 8)),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.before_latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(True)
        )
        self.mean_fc = nn.Linear(self.latent_dim, 32)
        self.var_fc = nn.Linear(self.latent_dim, 32)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(32, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_dim, self.before_latent_dim),
            nn.BatchNorm1d(self.before_latent_dim),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
#            nn.ConvTranspose1d(int(dim / 8), int(dim / 4), 3, stride=1),
            nn.ConvTranspose1d(dim, dim, 3, stride=1),
#            nn.BatchNorm1d(int(dim / 4)),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
#            nn.ConvTranspose1d(int(dim / 4), int(dim / 2), 2, stride=1),
            nn.ConvTranspose1d(dim, dim, 2, stride=1),
#            nn.BatchNorm1d(int(dim / 2)),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
#            nn.ConvTranspose1d(int(dim / 2), dim, 4, stride=1)
            nn.ConvTranspose1d(dim, dim, 4, stride=1)
        )


    def encode(self, x):
        output = self.encoder(x)
        output = output.view(-1, self.before_latent_dim)
        feat = self.encoder_fc(output)

        return self.mean_fc(feat), self.var_fc(feat), feat

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def decode(self, z):
        output = self.decoder_fc(z)
#        output = output.view(-1, int(self.dim / 8), (cfg.sample_length - 6))
        output = output.view(-1, self.dim , (cfg.sample_length - 6))
        output = self.decoder(output)
        return output

    def forward(self, x):
        mu, logvar, feat = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        regul = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, regul, feat
