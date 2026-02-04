import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, 2)  # 2 classes: real vs fake

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        x = x[:, 0]  # take the first frame only -> (B, C, H, W)

        x = self.features(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)

        return self.classifier(x)  # (B, 2)
