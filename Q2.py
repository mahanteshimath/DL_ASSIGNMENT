import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def download_frey_face():
    train_files = os.listdir('data/Train') if os.path.exists('data/Train') else []
    test_files = os.listdir('data/Test') if os.path.exists('data/Test') else []
    if not train_files and not test_files:
        print("Loading Frey Face dataset from .mat file...")
        mat_data = scipy.io.loadmat('data/frey_rawface.mat')
        images = mat_data['ff'].T.reshape(-1, 28, 20)
        images = images.astype(np.float32) / 255.0
        train_size = int(0.8 * len(images))
        train_images = images[:train_size]
        test_images = images[train_size:]
        print("Saving images to Train and Test directories...")
        for i, img in enumerate(train_images):
            img_uint8 = (img * 255).astype(np.uint8)
            Image.fromarray(img_uint8, mode='L').save(f'data/Train/train_{i:04d}.png')
        for i, img in enumerate(test_images):
            img_uint8 = (img * 255).astype(np.uint8)
            Image.fromarray(img_uint8, mode='L').save(f'data/Test/test_{i:04d}.png')
        print("Dataset split and saved to Train/Test directories")
    else:
        print("Loading existing images from Train/Test directories")
        train_images = []
        for filename in sorted(os.listdir('data/Train')):
            if filename.endswith('.png'):
                img = Image.open(os.path.join('data/Train', filename))
                img_array = np.array(img).astype(np.float32) / 255.0
                train_images.append(img_array)
        test_images = []
        for filename in sorted(os.listdir('data/Test')):
            if filename.endswith('.png'):
                img = Image.open(os.path.join('data/Test', filename))
                img_array = np.array(img).astype(np.float32) / 255.0
                test_images.append(img_array)
        images = np.concatenate([train_images, test_images], axis=0)
    print(f"Dataset shape: {images.shape}")
    return images

class FreyFaceDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image.astype(np.uint8), mode='L')
        if self.transform:
            image = self.transform(image)
        return image

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten_size = 128 * 3 * 2
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(size=(28, 20), mode='bilinear', align_corners=False)
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 3, 2)
        h = self.decoder_conv(h)
        return self.upsample(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    avg_loss = train_loss / len(train_loader.dataset)
    avg_bce = train_bce / len(train_loader.dataset)
    avg_kld = train_kld / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')
    return avg_loss, avg_bce, avg_kld

def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            test_bce += bce.item()
            test_kld += kld.item()
    avg_loss = test_loss / len(test_loader.dataset)
    avg_bce = test_bce / len(test_loader.dataset)
    avg_kld = test_kld / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')
    return avg_loss, avg_bce, avg_kld

def generate_samples(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        sample = model.decode(z).cpu()
        return sample

def visualize_reconstruction(model, test_loader, num_images=8):
    model.eval()
    with torch.no_grad():
        data_iter = iter(test_loader)
        data = next(data_iter)
        data = data.to(device)
        recon_batch, _, _ = model(data)
        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(min(num_images, len(data))):
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_batch[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig('reconstructions.png', dpi=150, bbox_inches='tight')
        plt.show()

def vary_latent_variables(model, base_z=None):
    model.eval()
    with torch.no_grad():
        if base_z is None:
            base_z = torch.randn(1, model.latent_dim).to(device)
        num_dims = min(6, model.latent_dim)
        variations = torch.linspace(-3, 3, 8)
        fig, axes = plt.subplots(num_dims, len(variations), figsize=(15, 2*num_dims))
        for dim in range(num_dims):
            for i, val in enumerate(variations):
                z = base_z.clone()
                z[0, dim] = val
                sample = model.decode(z)
                axes[dim, i].imshow(sample.cpu().squeeze(), cmap='gray')
                if i == 0:
                    axes[dim, i].set_ylabel(f'Dim {dim}')
                axes[dim, i].set_title(f'{val:.1f}')
                axes[dim, i].axis('off')
        plt.tight_layout()
        plt.savefig('latent_variations.png', dpi=150, bbox_inches='tight')
        plt.show()

def visualize_latent_space_2d(model, test_loader, num_samples=500):
    model.eval()
    all_mu = []
    with torch.no_grad():
        count = 0
        for data in test_loader:
            if count >= num_samples:
                break
            data = data.to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu().numpy())
            count += len(data)
        all_mu = np.concatenate(all_mu, axis=0)[:num_samples]
    plt.figure(figsize=(10, 8))
    plt.scatter(all_mu[:, 0], all_mu[:, 1], alpha=0.6)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Visualization of Latent Space')
    plt.savefig('latent_space_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Preparing Frey Face dataset...")
    images = download_frey_face()
    print(f"Total images: {images.shape[0]}")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = FreyFaceDataset(images[:int(0.8 * len(images))], transform=transform)
    test_dataset = FreyFaceDataset(images[int(0.8 * len(images)):], transform=transform)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = VAE(latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    train_losses = []
    test_losses = []
    num_epochs = 30
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_bce, train_kld = train(model, train_loader, optimizer, epoch)
        test_loss, test_bce, test_kld = test(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, f'vae_model_epoch_{epoch}.pth')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Generating random samples...")
    samples = generate_samples(model, 10)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualizing reconstructions...")
    visualize_reconstruction(model, test_loader)
    print("Varying latent variables...")
    vary_latent_variables(model)
    print("Creating 2D latent space visualization...")
    visualize_latent_space_2d(model, test_loader)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, 'vae_final_model.pth')
    print("Training completed and model saved!")
    print("Generated files:")
    print("- vae_final_model.pth")
    print("- loss_curve.png")
    print("- generated_samples.png")
    print("- reconstructions.png")
    print("- latent_variations.png")
    print("- latent_space_2d.png")

if __name__ == "__main__":
    main()
