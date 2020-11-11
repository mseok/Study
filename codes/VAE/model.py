import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, max_idx):
        super(VAE, self).__init__()
        self.enc_fc = nn.Linear(max_idx, 128, bias=False)
        self.encoder = nn.GRUCell(128, 128)

        self.mean_fc = nn.Linear(128, 128)
        self.var_fc = nn.Linear(128, 128)

        self.decoder = nn.GRUCell(128, 128)
        self.dec_fc = nn.Linear(128, max_idx, bias=False)

    def forward(self, onehot):
        batch_size, smiles_length = onehot.shape[:-1]
        # Encoder
        enc_onehot = self.enc_fc(onehot)
        enc_onehot = enc_onehot.view(-1, 128)
        enc_vector = self.encoder(enc_onehot)

        # Reparameterization Trick
        mu = self.mean_fc(enc_vector)
        sigma = self.var_fc(enc_vector)
        latent_vector = self.reparameterize(mu, sigma)

        # Decoder
        dec_vector = self.decoder(latent_vector)
        dec_onehot = self.dec_fc(dec_vector)
        dec_onehot = dec_onehot.view(batch_size, -1, dec_onehot.size(-1))

        # reconstruction loss
        # Decoder onehot and true onehot will be compared

        # KL Divergence loss
        kl_loss = - 0.5 * (1 + (sigma ** 2).log() - mu ** 2 - sigma ** 2)
        kl_loss = kl_loss.mean(-1)
        return dec_onehot, kl_loss
    
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn(std.size()).to(self.enc_fc.weight.device)
        latent_vector = eps.mul(std).add_(mu)
        return latent_vector
