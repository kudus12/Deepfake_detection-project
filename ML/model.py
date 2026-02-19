# model.py
# My CNN model: takes video frames (B,T,C,H,W) and outputs 2 classes (real/fake)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # My feature extractor (stronger than the old one)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
        )

        # My classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)
        I run CNN per-frame, then average across time.
        """
        B, T, C, H, W = x.shape

        # Merge batch + time so CNN sees images
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten

        logits = self.classifier(x)  # (B*T, 2)

        # Bring back time dimension
        logits = logits.view(B, T, -1)  # (B, T, 2)

        # Average frame predictions -> video prediction
        logits = logits.mean(dim=1)  # (B, 2)

        return logits
