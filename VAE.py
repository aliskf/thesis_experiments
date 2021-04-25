import torch
from torch import nn
from torch.nn import functional as F



class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()
        z = 32 # Latent dimensionality
        
        self.fc1 = nn.Linear(150*150, 1000)
        self.fc21 = nn.Linear(1000, z) # mean
        self.fc22 = nn.Linear(1000, z) # std
        self.fc3 = nn.Linear(z, 1000)
        self.fc4 = nn.Linear(1000, 150*150)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # don't forget forward pass re-index
        mu, logvar = self.encode(x.view(-1, 150*150))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

