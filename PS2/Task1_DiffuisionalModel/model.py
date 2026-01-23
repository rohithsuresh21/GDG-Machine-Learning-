import torch
import torch.nn as nn
from utils import time_embedding

class SimpleUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU()
        )

        # Channel projection layers for time embedding
        self.time_proj1 = nn.Linear(128, 64)
        self.time_proj2 = nn.Linear(128, 128)

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Time embedding
        t_emb = time_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)  # (batch, 128)

        # Project time embedding to match channels
        t1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)  # (batch, 64, 1, 1)
        t2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)

        # Forward
        x1 = self.relu(self.conv1(x))           # (batch, 64, H, W)
        x2 = self.relu(self.conv2(x1 + t1))     # channel match 64 
        x3 = self.relu(self.conv3(x2 + t2))     # channel match 128 
        out = self.conv4(x3)

        return out
