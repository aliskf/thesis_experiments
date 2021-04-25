import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE import VAE
from dataloader import val_loader, train_loader,epochs
from tqdm import tqdm

import os
import shutil
import random
random.seed(5)

import matplotlib

import matplotlib.pyplot as plt
plt.style.use('ggplot')





def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 150*150), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate(evaluate_data=val_loader):
    
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(evaluate_data):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar).item()
            
            if i==0:
                n = min(data.size(0), 16)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 3, 150, 150)[:n]])
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.jpg', nrow=n)

    val_loss /= len(evaluate_data.dataset)
    return val_loss


def sample_latent_space(epoch):
    with torch.no_grad():
        sample = torch.randn(64, 32).to(device)
        sample = model.decode(sample).cpu() # Switch to CPU in order to save the image
        save_image(sample.view(64, 1, 150, 150), 'results/sample_' + str(epoch) + '.jpg')


def train(epoch):

    model.train()
    train_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)
    for data, _ in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(data))})

    average_train_loss = train_loss / len(train_loader.dataset)
    tqdm.write('Training set loss (average, epoch {:03d}): {:.3f}'.format(epoch, average_train_loss))
    val_loss = evaluate(val_loader)
    tqdm.write('\t\t\t\t====> Validation set loss: {:.3f}'.format(val_loss))

    train_losses.append(average_train_loss)
    val_losses.append(val_loss)
    
    if epoch%50==0:
        torch.save(model.state_dict(), f'Models/epoch_{epoch}.model')

train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    train(epoch)
    sample_latent_space(epoch)
     
np.savetxt('Models/training_losses.txt', np.array(train_losses), delimiter='\n')
np.savetxt('Models/validation_losses.txt', np.array(val_losses), delimiter='\n')


train()