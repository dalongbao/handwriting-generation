import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from data import get_dataset 
from model import Discriminator, Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
random.seed(69)
torch.manual_seed(69)
np.random.seed(69)

dataset = get_dataset()
random_noise_vector = torch.randn(16, 100, device=device) * 2 - 1
os.makedirs('weights', exist_ok=True)

def show_image(images, label):
    image = images[0]  # Shape: (1, 180, 13)
    image = image.squeeze(0)  # Shape: (180, 13)
    plt.figure(figsize=(5, 5))
    plt.imshow(image.cpu().numpy(), cmap='binary')
    plt.title(f"Label: {label[0]}")
    plt.axis('off')
    plt.show()

def show_generated_image(images, is_generated=False, num_to_show=4):
    print(f"Input tensor shape: {images.shape}")
    
    # Move the tensor to CPU and convert to numpy
    images = images.cpu().detach().numpy()
    
    # Reshape the flattened output to the correct image dimensions (30x400)
    images = images.reshape(-1, 30, 400)
    
    print(f"Reshaped images shape: {images.shape}")
    
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_to_show, images.shape[0])):
        ax = axes[i]
        im = ax.imshow(images[i], aspect='auto', cmap='Greys', origin='lower')
        ax.set_title(f"Generated Image {i+1}" if is_generated else f"Real Image {i+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

"""Hyperparameters"""
lr = 2e-3
num_epochs = 500
save_interval = 50

"""Model"""
discriminator = Discriminator(12000).to(device)
generator = Generator(100, 12000).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()

for e in range(num_epochs):
    for real_images, _ in dataset:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        """Discriminator training"""
        discriminator.zero_grad()

        d_output_real = discriminator(real_images)
        d_loss_real = loss_fn(d_output_real, real_labels)
        
        noise = torch.randn(batch_size, 100, device=device) * 2 - 1
        fake_images = generator(noise)
        d_output_fake = discriminator(fake_images.detach())
        d_loss_fake = loss_fn(d_output_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        generator.zero_grad()
        d_output_fake = discriminator(fake_images)
        g_loss = loss_fn(d_output_fake, real_labels)  # We want the discriminator to think these are real
        g_loss.backward()
        g_optimizer.step()

    # Print losses
    print(f"Epoch [{e}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Generate and save sample images
    if e % 100 == 0 and e != 0:
        generator.eval()
        with torch.no_grad():
            fake_images = generator(random_noise_vector)
            # Save or display fake_images
            show_generated_image(fake_images, is_generated=True)
        generator.train()

    if (e + 1) % save_interval == 0 or e == num_epochs - 1:
            torch.save(generator.state_dict(), f'weights/generator_epoch_{e+1}.pth')
            torch.save(discriminator.state_dict(), f'weights/discriminator_epoch_{e+1}.pth')
            print(f"Saved model weights at epoch {e+1}")

# Save final weights after training
torch.save(generator.state_dict(), 'weights/generator_final.pth')
torch.save(discriminator.state_dict(), 'weights/discriminator_final.pth')
print("Saved final model weights")
