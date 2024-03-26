import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset_path = './data/img_align_celeba/'

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
        )
        
        self.z_mean = nn.Linear(256, 64)
        self.z_log_var = nn.Linear(256, 64)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        encoded = self.encoder(x)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        z = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(z)
        return decoded, z_mean, z_log_var



vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_function = nn.MSELoss()

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        reconstructed, z_mean, z_log_var = vae(data)
        
        reconstruction_loss = loss_function(reconstructed, data)
        kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = reconstruction_loss + kl_divergence
        
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")


with torch.no_grad():
    z = torch.randn(1, 64)
    generated_image = vae.decoder(z)
    plt.imshow(generated_image.squeeze(0).permute(1, 2, 0))
    plt.show()
