import torch
import math

def _pos_encoding(time_idx, output_dim, device='cpu'):
    """
    Computes the positional encoding vector for a single time step (t).
    Uses the standard Transformer sinusoidal encoding method.
    """
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    # The division term controls the frequency of the sine/cosine waves
    div_term = torch.exp(i / D * math.log(10000))

    # Apply sin to even indices
    v[0::2] = torch.sin(t / div_term[0::2])
    # Apply cos to odd indices
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    """
    Computes positional encoding for a batch of time steps.
    """
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    
    # Iterate over the batch of timesteps and call the single-step helper
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i].float(), output_dim, device)
        
    return v