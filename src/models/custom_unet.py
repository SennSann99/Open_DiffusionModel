# unet.py

import torch
from torch import nn
import torch.nn.functional as F


from src.utils.pos_encoding import pos_encoding


class ConvBlock(nn.Module):
    """
    A block containing two convolutional layers with Batch Normalization
    and ReLU activation, incorporating a Time Step Embedding.
    """
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # MLP for time embedding (v)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        # Process time embedding and broadcast to match feature map shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        # Add time-conditioned vector to the feature map before convolutions
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    """
    A simple U-Net model adapted for Diffusion Models (time-conditioned).
    """
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Downsampling path
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.maxpool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ConvBlock(128, 256, time_embed_dim)

        # Upsampling path
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim) # Skip connection concat adds channels
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)

        # Output layer
        self.out = nn.Conv2d(64, in_ch, 1)

    def forward(self, x, timesteps):
        # 1. Compute time embedding (v)
        # timesteps: (N,) tensor of time indices
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        # 2. Downsampling
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        # 3. Bottleneck
        x = self.bot1(x, v)

        # 4. Upsampling (with skip connections)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1) # Concatenate with skip connection from down2
        x = self.up2(x, v)
        
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1) # Concatenate with skip connection from down1
        x = self.up1(x, v)
        
        # 5. Output
        x = self.out(x)
        return x