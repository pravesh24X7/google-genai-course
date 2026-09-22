# In this first lab we'll use the `Unet` from segmentation_models_pytorch module of python.
# ou the MNIST dataset.
# later I'll extend this to work with any image dataset.


import torch
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))])

train = datasets.MNIST(
    root="./data/",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train, batch_size=32, shuffle=True)

test = datasets.MNIST(
    root="./data/",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test, shuffle=False, batch_size=32)

# plotting one single image to check how it looks

image, label = train[0]
img = image.permute(1, 2, 0)        # because for 3D image `imshow` method expect color channels as last dimension
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()

# loading the Unet model for creating map, first trying this on one single image.
model = smp.Unet("resnet50", encoder_weights='imagenet', in_channels=3, classes=3, activation=None)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=DEVICE)

# Now that we know how Unet works let's integrate this in diffusion loop.
# Training loop    

T = 10
N = 20
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)


for epoch in range(N):
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device=DEVICE), labels.to(device=DEVICE)

        optimizer.zero_grad()
        t = np.random.randint(0, T)

        epsilon = torch.randn_like(images)
        alpha_bar_t = float(alpha_bars[t])
        images_t = np.sqrt(alpha_bar_t) * images + np.sqrt(1-alpha_bar_t) * epsilon

        epsilon_predicted = model(images_t)

        loss = nn.functional.mse_loss(epsilon_predicted, epsilon)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{N}  Loss: {loss.item():.4f}")


# Testing loop
model.eval()

image_shape = (32, 3, 32, 32)
with torch.no_grad():
    
    images_t = torch.randn_like(image_shape)
    generated = []
    starting_noise = images_t[0].clone()        # just for later comparison

    images_t = images_t.to(device=DEVICE)
    
    for t in reversed(range(T)):
        beta_t = float(betas[t])
        alpha_t = float(alphas[t])
        alpha_bar_t = float(alpha_bars[t])

        epsilon_predicted = model(images_t)
        mean = (1/np.sqrt(alpha_t)) * (images_t - (beta_t / np.sqrt(1-alpha_bar_t))*epsilon_predicted)
        
        if t>0:
            z = torch.randn_like(images)
            images_t = mean + np.sqrt(beta_t) * z
        else:
            images_t = mean

        mean = (1/np.sqrt(alpha_t)) * (images_t - (beta_t / np.sqrt(1-alpha_bar_t))*epsilon_predicted)
        
        if t>0:
            z = torch.randn_like(images)
            images_t = mean + np.sqrt(beta_t) * z
        else:
            images_t = mean
    x0_generated = images_t

# How to display the generated images along with starting image, Try using the plt.subplot for better comparsion
def to_displayable(tensor):
    # reverse normalize from [-1,1] back to [0,1], then to HWC
    img = tensor.cpu().clamp(-1, 1) * 0.5 + 0.5
    return img.permute(1, 2, 0).numpy()

n = 4    # how many images to show side by side
fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

for i in range(n):
    # top row: starting noise
    axes[0, i].imshow(to_displayable(starting_noise), cmap='gray')
    axes[0, i].set_title("Start (noise)")
    axes[0, i].axis("off")

    # bottom row: generated image
    axes[1, i].imshow(to_displayable(x0_generated[i]), cmap='gray')
    axes[1, i].set_title("Generated")
    axes[1, i].axis("off")

plt.suptitle("Diffusion Inference — Noise → Generated", fontsize=13)
plt.tight_layout()
plt.show()

