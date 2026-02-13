import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import DeepfakeVideoDataset
from model import SimpleCNN

# Paths (use your working ones)
REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\real\original"
FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\fake\Deepfakes"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load dataset
    dataset = DeepfakeVideoDataset(REAL_DIR, FAKE_DIR, frame_limit=10)

    # 2) DataLoader (batches)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3) Model
    model = SimpleCNN().to(device)

    # 4) Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5) Training loop (1 epoch test)
    model.train()
    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device)   # (B, T, C, H, W)
        labels = labels.to(device)   # (B,)

        # Forward
        outputs = model(frames)      # (B, 2)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

        # Just test first 5 batches (so it doesn't run forever)
        if batch_idx == 4:
            break

    print("Training step completed ✅")

if __name__ == "__main__":
    main()
