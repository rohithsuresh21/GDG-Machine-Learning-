import torch                                    # for tensor operations
import torch.nn.functional as F                 # for loss functions
from torch.utils.data import DataLoader         # for loading datasets
from torchvision import datasets, transforms    # for dataset and transformations
from model import SimpleUNet                    # import the model
from diffusion import forward_diffusion, T      # import diffusion process and total steps
from tqdm import tqdm                           # progress bar for training visibility

device = "cuda" if torch.cuda.is_available() else "cpu"         # Set device to GPU if available

print("Using device:", device)                  # show device
print("Loading dataset...")                     # logging

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])                                              # Normalize to [-1, 1] and convert to tensor

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)        # Load MNIST dataset
loader = DataLoader(dataset, batch_size=64, shuffle=True)          # DataLoader for batching

print("Dataset loaded successfully")             # logging

model = SimpleUNet().to(device)      # Initialize the model and move to device 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)     # Adam optimizer

epochs = 5  # low epochs 

print("Training started...")                     # logging

for epoch in range(epochs):              # Training loop
    pbar = tqdm(loader)                  # progress bar per epoch
    for x0, _ in pbar:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),)).to(device)

        xt, noise = forward_diffusion(x0, t)        # Get noisy image at time step t
        noise_pred = model(xt, t)                   # Predict the noise using the model

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")   # live loss display

    print(f"Epoch {epoch+1} completed | Final Loss: {loss.item():.4f}")      # epoch summary

torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved as diffusion_model.pth")            # confirmation
print("Training completed!")                       # logging