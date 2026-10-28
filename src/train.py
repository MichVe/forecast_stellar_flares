from __future__ import annotations

import logging

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def make_classifier_loaders(X, y, batch_size=64, val_frac=0.1, test_frac=0.1, seed=42):
    """80/10/10 train/val/test split, matching the paper's Results section."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_frac + test_frac, stratify=y, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed
    )

    def loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader(X_train, y_train, True), loader(X_val, y_val, False), loader(X_test, y_test, False)


def make_forecast_loaders(X, y, batch_size=64, val_frac=0.1, test_frac=0.1, seed=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=val_frac + test_frac, random_state=seed)
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=rel_test, random_state=seed)

    def loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_).unsqueeze(-1), torch.from_numpy(y_).unsqueeze(-1))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader(X_train, y_train, True), loader(X_val, y_val, False), loader(X_test, y_test, False)


def train_classifier(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_train = train_loader.dataset.tensors[1]
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    logger.info("classifier class balance: %d pos / %d neg -> pos_weight=%.2f", int(n_pos), int(n_neg), pos_weight.item())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info("classifier epoch %d: train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_forecaster(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu", patience=None):
    """`patience`: stop after this many epochs without a validation-loss improvement
    (None = always run all epochs, as in v1)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, horizon=yb.shape[1])
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, horizon=yb.shape[1])
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info("forecaster epoch %d: train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                logger.info("forecaster early stop at epoch %d (no val improvement for %d epochs)", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
