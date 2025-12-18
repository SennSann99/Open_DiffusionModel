import torch.nn as nn
from diffusers import UNet2DModel

class hf_unet(nn.Module): # Inherit from nn.Module
    def __init__(self, img_size):
        super().__init__() # Initialize the parent class
        self.model = UNet2DModel(
            
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
    
    def forward(self, x, t): # Add a forward method
        return self.model(x, t)