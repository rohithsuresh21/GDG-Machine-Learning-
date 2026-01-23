import torch
from model import SimpleUNet
from diffusion import betas, alphas, alpha_bar, T
from torchvision.utils import save_image               # for saving generated images

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUNet().to(device)         # Initialize the model and move to device
model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))           # Load the trained model weights
model.eval()           # Set the model to evaluation mode

x = torch.randn((1, 1, 28, 28)).to(device)     # Start from random noise with 1 channel and 28x28 size

for t in reversed(range(T)):                # Reverse diffusion process
    t_tensor = torch.tensor([t]).to(device)   # Current time step tensor
    eps_pred = model(x, t_tensor)             # Predict the noise at time step t

    alpha = alphas[t].to(device)                # Get alpha at time step t
    ab = alpha_bar[t].to(device)                # Get alpha_bar at time step t
    beta = betas[t].to(device)                  # Get beta at time step t

    x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - ab)) * eps_pred)

    if t > 0:
        x = x + torch.sqrt(beta) * torch.randn_like(x)

save_image((x + 1) / 2, "generated.png")
print("Image generated -> generated.png")
