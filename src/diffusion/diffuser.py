# diffuser.py

import torch
import math
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F


class Diffuser:
    """
    Manages the forward (add_noise) and reverse (denoise/sample) processes
    of the DDPM.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define schedules: beta, alpha, alpha_bar
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Ensure alpha_bar_prev calculation can handle t=1
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0) # [1.0, alpha_bar_1, alpha_bar_2, ...]

    def add_noise(self, x_0, t):
        """
        Forward process: Perturbs data x_0 at time t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1 
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        return x_t, noise

    def denoise(self, model, x, t):
        """
        Reverse process step: Predicts x_{t-1} from x_t using model prediction.
        Model predicts the noise (eps).
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars_prev[t_idx] # Corrected to use the padded version for t=1

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t) # Predict the noise at time t
        model.train()
        
        # Calculate mean (mu) and variance (std^2) for the reverse step q(x_{t-1} | x_t)
        # mu_tilde(x_t, eps)
        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        
        # sigma_tilde^2 * I
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        
        # Sample x_{t-1} ~ N(mu_tilde, sigma_tilde^2 * I)
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # No noise at the final step (t=1)
        
        return mu + noise * std
        
    def reverse_to_img(self, x):
        """Converts model output back to a displayable PIL image."""
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        """Performs the full reverse/sampling process."""
        batch_size = x_shape[0]
        # Start from pure noise
        x = torch.randn(x_shape, device=self.device)

        # Iterate from T down to 1
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images