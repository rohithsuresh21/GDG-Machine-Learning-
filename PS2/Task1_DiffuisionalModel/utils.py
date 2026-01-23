# Time Embedding Utils
import torch           # For tensor operations
import math            # For mathematical functions

def time_embedding(t, dim):           # Generate sinusoidal time embeddings
    half = dim // 2                   # Half the embedding dimension
    emb = math.log(10000) / (half - 1)   # Scaling factor for frequencies
    emb = torch.exp(torch.arange(half) * -emb).to(t.device)       # Frequency terms
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)         # Concatenate sine and cosine embeddings
    return emb
