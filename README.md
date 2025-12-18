# Diffusion Model Implementation

A modular and easy-to-understand implementation of a Diffusion Model. This repository breaks down the diffusion process into clear components: the forward/reverse logic, the U-Net architecture, and the training loop.



## 🛠 Features

- **Modular Design**: Separate modules for the model architecture, diffusion logic, and utilities.
- **U-Net Backbone**: Implementation of a U-Net with positional encodings for time-step embeddings.
- **Modern Tooling**: Managed with `uv` for lightning-fast dependency management and reproducible environments.
- **Interactive Demos**: Includes a Jupyter notebook for step-by-step visualization of the denoising process.

## 📂 Project Structure

```text
.
├── src/
│   ├── diffusion/      # Forward (noise) and Reverse (denoise) logic
│   │   └── diffuser.py
│   ├── models/         # Neural network architectures
│   │   └── unet.py
│   ├── training/       # Training scripts and loss functions
│   │   └── main_train.py
│   ├── utils/          # Positional encodings and helper functions
│   │   └── pos_encoding.py
│   └── __init__.py
├── notebooks/          # Experimentation and visualization
├── outputs/            # Saved plots and generated images
├── main.py             # Entry point for training/inference
└── pyproject.toml      # Project configuration and dependencies