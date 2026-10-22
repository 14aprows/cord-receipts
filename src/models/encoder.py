import torch 
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(
        self, hidden_size=256
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16,16))
        )

        self.position = nn.Parameter(
            torch.randn(1, 256, hidden_size) * 0.02
        )

    def forward(self, x):
        # [B, H, 16, 16]
        x = self.features(x)
        # [B, H, 256]
        x = x.flatten(2)
        # [B, 256, H]
        x = x.transpose(1,2)

        return x + self.position