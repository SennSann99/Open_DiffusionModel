# src/training/main_train.py
import sys
import os

# Get the path to the project root (Go up two levels: training -> src -> Root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

# --- NEW IMPORT ---
from diffusers import UNet2DModel 
from src.diffusion.diffuser import Diffuser

# --- 1. Configuration Constants ---
img_size = 28 # MNIST default
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-4 # Lower learning rate is usually better for these models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Utility Functions ---
def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(images): break
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()

def train():
    print(f"--- Starting DDPM Training with HF Diffusers on Device: {device} ---")

    # --- 3. Data Loading ---
    # Note: If you want to use a model pretrained on CIFAR/ImageNet, you must resize to 32x32 or 64x64.
    # For a fresh 'better' architecture on MNIST, 28x28 is fine.
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((32, 32)), # Uncomment if using a 32x32 specific config
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for diffusers
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 4. Model Setup (Replaced Custom UNet) ---
    diffuser = Diffuser(num_timesteps, device=device)
    
    # Using a high-quality UNet configuration
    model = UNet2DModel(
        sample_size=img_size,  # target image resolution
        in_channels=1,         # MNIST has 1 channel
        out_channels=1,        # Output matches input
        layers_per_block=2,    # How many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 128), # Channel counts for each block
        down_block_types=(
            "DownBlock2D",      # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",   
            "UpBlock2D",      
        ),
    )
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # --- 5. Training Loop ---
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            x = images.to(device)
            
            t = torch.randint(0, num_timesteps, (len(x),), device=device).long() # diffusers prefers 0-indexed longs

            x_noisy, noise = diffuser.add_noise(x, t)
            
            # --- MODEL CHANGE HERE ---
            # HF models return a tuple/object. We need .sample to get the tensor.
            output = model(x_noisy, t)
            noise_pred = output.sample 
            
            loss = F.mse_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Average Loss: {loss_avg:.4f}')

    print("Training finished.")

    # --- 6. Post-Training Visualization ---
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.show()

    print("Generating final samples...")
    # Ensure your diffuser.sample() function can handle the model.
    # You might need to adjust your diffuser.sample to do: predict = model(x, t).sample
    images = diffuser.sample(model, x_shape=(20, 1, img_size, img_size))
    show_images(images)

if __name__ == '__main__':
    train()