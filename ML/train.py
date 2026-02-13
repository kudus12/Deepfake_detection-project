import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import DeepfakeVideoDataset
from model import SimpleCNN

# 1) Paths (keep same as your working ones)
REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\real\original"
FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\fake\Deepfakes"

# 2) Settings
FRAME_LIMIT = 10
BATCH_SIZE = 2

# 3) Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 4) Dataset + DataLoader
dataset = DeepfakeVideoDataset(REAL_DIR, FAKE_DIR, frame_limit=FRAME_LIMIT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5) Model
model = SimpleCNN().to(device)
print("Model ready ✅")

# Quick sanity check: one batch through model (NO training yet)
frames, labels = next(iter(loader))
frames, labels = frames.to(device), labels.to(device)

outputs = model(frames)
print("Batch input:", frames.shape)
print("Batch output:", outputs.shape)
print("Batch labels:", labels)
