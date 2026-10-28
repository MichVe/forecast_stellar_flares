from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StellaCNN(nn.Module):
    """Flare / non-flare CNN classifier reproducing the Stella architecture
    (Feinstein et al. 2020) as diagrammed in Figure 2b of the paper:
    Conv-7-16 -> MaxPool2 -> Dropout0.1 -> Conv-3-64 -> MaxPool2 -> Dropout0.1
    -> Flatten -> Dense-32 -> Dropout0.1 -> output.
    """

    def __init__(self, window: int = 200, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout)
        flat_dim = 64 * (window // 4)
        self.fc1 = nn.Linear(flat_dim, 32)
        self.drop3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop3(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # logits, use with BCEWithLogitsLoss
