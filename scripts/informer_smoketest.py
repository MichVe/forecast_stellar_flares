# -*- coding: utf-8 -*-
"""Smoke test: verify the Informer wiring (shapes, forward/generate) works on a tiny
synthetic batch before committing to a full training run."""
import torch
from transformers import InformerConfig, InformerForPrediction

LOOKBACK, HORIZON = 150, 50
LAGS = [1, 2, 3, 4, 5]
MAX_LAG = max(LAGS)
CONTEXT_LENGTH = LOOKBACK - MAX_LAG  # so past_values length == LOOKBACK exactly

config = InformerConfig(
    prediction_length=HORIZON,
    context_length=CONTEXT_LENGTH,
    lags_sequence=LAGS,
    num_time_features=1,
    num_static_categorical_features=0,
    num_static_real_features=0,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    dropout=0.1,
    scaling="mean",
)
model = InformerForPrediction(config)

B = 4
past_values = torch.randn(B, LOOKBACK)
past_time_features = (torch.arange(LOOKBACK).float() / LOOKBACK).view(1, -1, 1).repeat(B, 1, 1)
past_observed_mask = torch.ones(B, LOOKBACK)
future_values = torch.randn(B, HORIZON)
future_time_features = (torch.arange(LOOKBACK, LOOKBACK + HORIZON).float() / LOOKBACK).view(1, -1, 1).repeat(B, 1, 1)

out = model(
    past_values=past_values,
    past_time_features=past_time_features,
    past_observed_mask=past_observed_mask,
    future_values=future_values,
    future_time_features=future_time_features,
)
print("training forward OK, loss =", out.loss.item())

model.eval()
with torch.no_grad():
    gen = model.generate(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        future_time_features=future_time_features,
    )
print("generate() OK, sequences shape:", gen.sequences.shape)  # (B, num_samples, HORIZON)
point = gen.sequences.mean(dim=1)
print("point forecast shape:", point.shape)
