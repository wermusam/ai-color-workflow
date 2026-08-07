"""A small CNN that predicts a CDL grade (10 numbers) from an image."""

import torch
import torch.nn as nn


class CdlNet(nn.Module):
    """Predicts a CDL grade (slope, offset, power, saturation) from an image."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict 10 numbers per image: slope(3), offset(3), power(3), saturation(1).

        The numbers are squashed into sensible ranges so slope and power stay
        positive, which keeps the grade math from blowing up.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        raw = self.fc(x)

        slope = torch.sigmoid(raw[:, 0:3]) * 2.0
        offset = torch.tanh(raw[:, 3:6]) * 0.5
        power = torch.sigmoid(raw[:, 6:9]) * 2.0 + 0.1
        saturation = torch.sigmoid(raw[:, 9:10]) * 2.0
        return torch.cat([slope, offset, power, saturation], dim=1)
