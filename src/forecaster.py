from __future__ import annotations

import torch
import torch.nn as nn


class ETSformerForecaster(nn.Module):
    """The paper's proposed forecaster: exponential-smoothing-attention Transformer
    (Woo et al. 2022), used here to predict the future course of the light curve
    from a lookback window (Section 5/6 of the paper)."""

    def __init__(self, model_dim: int = 64, layers: int = 2, heads: int = 4, K: int = 4, dropout: float = 0.2):
        super().__init__()
        from etsformer_pytorch import ETSFormer

        self.net = ETSFormer(
            time_features=1,
            model_dim=model_dim,
            layers=layers,
            heads=heads,
            K=K,
            dropout=dropout,
        )

    def forward(self, lookback: torch.Tensor, horizon: int) -> torch.Tensor:
        # lookback: (batch, lookback_len, 1) -> (batch, horizon, 1)
        return self.net(lookback, num_steps_forecast=horizon)


class LSTMForecaster(nn.Module):
    """Classical recurrent baseline, representing the CNN/LSTM literature the paper
    contrasts against (e.g. Piana et al. 2024's Recurrent-Convolutional Network)."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, lookback: torch.Tensor, horizon: int) -> torch.Tensor:
        _, (h, c) = self.lstm(lookback)
        step_input = lookback[:, -1:, :]
        outputs = []
        for _ in range(horizon):
            out, (h, c) = self.lstm(step_input, (h, c))
            step = self.head(out)
            outputs.append(step)
            step_input = step
        return torch.cat(outputs, dim=1)


class NaivePersistenceForecaster:
    """Classical baseline (Section 2's "statistical prediction" family): repeats the
    last observed lookback value across the whole horizon."""

    def __call__(self, lookback: torch.Tensor, horizon: int) -> torch.Tensor:
        return lookback[:, -1:, :].repeat(1, horizon, 1)

    def eval(self):
        return self
