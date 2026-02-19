# train_loop.py
# My full training loop (real vs fake) using:
# - dataset_loader.py (DeepfakeVideoDataset with FACE crops)
# - model.py (SimpleCNN)
#
# It trains + validates, prints accuracy each epoch, and saves the best model.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_loader import DeepfakeVideoDataset
from model import SimpleCNN


# ------------------------
# My dataset paths
# ------------------------
REAL_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\real\original"
FAKE_DIR = r"C:\Users\kudus\OneDrive - Atlantic TU\4th year\Research in Computing\full-progress-code\Dataset\fake\Deepfakes"


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).sum().item(), labels.size(0)


def main():
    # ------------------------
    # Device
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------
    # Dataset
    # ------------------------
    dataset = DeepfakeVideoDataset(REAL_DIR, FAKE_DIR, frame_limit=16, face_size=224)
    print("Total videos:", len(dataset))

    # ------------------------
    # Split (80/20)
    # ------------------------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # ------------------------
    # DataLoaders
    # ------------------------
    # If it feels slow, set batch_size=1
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    # ------------------------
    # Model
    # ------------------------
    model = SimpleCNN(num_classes=2).to(device)

    # ------------------------
    # Loss + Optimizer + Scheduler
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ------------------------
    # Training settings
    # ------------------------
    epochs = 15
    best_val_acc = 0.0

    os.makedirs("saved_models", exist_ok=True)
    save_path = "saved_models/best_model.pth"

    # ------------------------
    # Loop
    # ------------------------
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # ===== TRAIN =====
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for frames, labels in train_loader:
            frames = frames.to(device)   # (B,T,C,H,W)
            labels = labels.to(device)   # (B,)

            logits = model(frames)       # (B,2)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            c, t = accuracy_from_logits(logits, labels)
            train_correct += c
            train_total += t

        train_loss = train_loss_sum / train_total
        train_acc = (train_correct / train_total) * 100

        # ===== VALIDATION =====
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)

                logits = model(frames)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                c, t = accuracy_from_logits(logits, labels)
                val_correct += c 
                val_total += t

        val_loss = val_loss_sum / val_total
        val_acc = (val_correct / val_total) * 100

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model: {save_path} (Val Acc {best_val_acc:.2f}%)")

        scheduler.step()

    print("\nDone ✅")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Best model saved at:", save_path)


if __name__ == "__main__":
    main()
