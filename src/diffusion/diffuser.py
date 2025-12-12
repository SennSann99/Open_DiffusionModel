import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

class Diffuser:
    """
    Manages the forward (add_noise) and reverse (denoise/sample) processes
    of the DDPM. 
    Updated for compatibility with Hugging Face Diffusers models.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define schedules: beta, alpha, alpha_bar
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # alpha_bars_prev[t] corresponds to alpha_bar_{t-1}
        # For t=0, previous alpha_bar is 1.0
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

    def add_noise(self, x_0, t):
        """
        Forward process: Perturbs data x_0 at time t.
        Now supports 0-based indexing (t=0 to T-1).
        """
        # Direct indexing (no t-1)
        alpha_bar = self.alpha_bars[t]
        
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        return x_t, noise

    def denoise(self, model, x, t):
        """
        Reverse process step: Predicts x_{t-1} from x_t.
        """
        # Direct indexing (no t-1)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars_prev[t]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            # --- FIX: Handle Hugging Face Output ---
            output = model(x, t)
            # HF models return an object with a .sample attribute
            if hasattr(output, 'sample'):
                eps = output.sample
            else:
                eps = output
        model.train()
        
        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        
        noise = torch.randn_like(x, device=self.device)
        
        # No noise at the very first step (t=0)
        noise[t == 0] = 0 
        
        return mu + noise * std
        
    def reverse_to_img(self, x):
        """Converts model output back to a displayable PIL image."""
        x = (x + 1) / 2 # Un-normalize from [-1, 1] to [0, 1]
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        """Performs the full reverse/sampling process."""
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        # Loop from T-1 down to 0
        for i in tqdm(range(self.num_timesteps - 1, -1, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images