# src/training/main_train.py

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

# Import modular components
from src.models.unet import UNet
from src.diffusion.diffuser import Diffuser


# --- 1. Configuration Constants ---
img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 2. Utility Functions (Moved from Notebook) ---

def show_images(images, rows=2, cols=10):
    """Displays a batch of images in a grid."""
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()


def train():
    """Main function to setup and run the diffusion model training loop."""
    print(f"--- Starting DDPM Training on Device: {device} ---")

    # --- 3. Data Loading ---
    preprocess = transforms.ToTensor()
    # Data is downloaded to the local 'data' directory
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 4. Model and Diffuser Setup ---
    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # --- 5. Training Loop ---
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        # Optional: Generate samples every epoch (uncomment below if desired)
        # print("Generating intermediate samples...")
        # images = diffuser.sample(model, x_shape=(20, 1, img_size, img_size))
        # show_images(images)

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            x = images.to(device)
            
            # Sample random time steps
            t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

            # Forward process: Add noise
            x_noisy, noise = diffuser.add_noise(x, t)
            
            # Reverse process: Predict the noise
            noise_pred = model(x_noisy, t)
            
            # Loss calculation (MSE between predicted noise and actual noise)
            loss = F.mse_loss(noise, noise_pred)

            # Optimization step
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Average Loss: {loss_avg:.4f}')

    print("Training finished.")

    # --- 6. Post-Training Visualization ---
    
    # Plot losses
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    # Generate final samples
    print("Generating final samples...")
    images = diffuser.sample(model, x_shape=(20, 1, img_size, img_size))
    show_images(images)


if __name__ == '__main__':
    train()