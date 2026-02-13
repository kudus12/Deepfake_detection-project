import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor (per frame)
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

        # Final classifier
        self.classifier = nn.Linear(64, 2)  # real / fake

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)
        """

        B, T, C, H, W = x.shape

        # merge batch and time
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = x.view(B * T, -1)  # flatten

        # restore batch dimension
        x = x.view(B, T, -1)

        # average frames
        x = x.mean(dim=1)

        # classify
        out = self.classifier(x)
        return out
