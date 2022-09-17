import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, in_dim),
            nn.Sigmoid()
        )
        self.device = device
    
    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y

    def get_latent(self, x):
        with torch.no_grad():
            latent = self.encoder(x)
        return latent
