import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from vae_model import VAE  # Make sure to import the VAE class correctly
import os

# Data loading
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
dataset_path = '/data/img_align_celeba/'
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model, optimizer, and loss function setup
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# TensorBoard and checkpoint setup
writer = SummaryWriter('runs/experiment_name')
checkpoint_dir = 'runs/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def save_sample_images(epoch, fixed_z, folder='runs/images', num_images=10):
    with torch.no_grad():
        samples = vae.decoder(fixed_z).cpu()
        os.makedirs(folder, exist_ok=True)
        save_image(samples, f'{folder}/epoch_{epoch}.png', nrow=int(num_images ** 0.5))

# Training loop
num_epochs = 10
fixed_z = torch.randn(64, 64)  # Adjust the size (64 here) to match your latent dimension

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        reconstructed, z_mean, z_log_var = vae(data)
        
        reconstruction_loss = loss_function(reconstructed, data)
        kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = reconstruction_loss + kl_divergence

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

    average_loss = running_loss / len(dataloader)
    writer.add_scalar('training loss', average_loss, epoch)
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    torch.save(vae.state_dict(), checkpoint_path)

    if epoch % 1 == 0:
        save_sample_images(epoch, fixed_z)

writer.close()
