"""Training loop for the forecaster.

The objective is KL against module 1's template mixtures: the model is asked to
predict the next day's mixture from the days before it. Only days in the chosen
split contribute, so validation days never enter the training gradient.
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.dataset import to_device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    split_id: int | None = None,
) -> float:
    """One pass over the loader. Returns mean KL per scored day (NaN if none).

    Batches whose target days all fall outside ``split_id`` contribute nothing,
    so an empty batch cannot poison the average.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_days = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    for batch in loader:
        batch = to_device(batch, device)
        target = batch["theta"][:, 1:]
        valid = target.sum(-1) > 0
        if split_id is not None:
            valid = valid & (batch["split"][:, 1:] == split_id)
        n_days = int(valid.sum())
        if n_days == 0:
            continue

        with context:
            logits = model(
                batch["arrival"], batch["departure"], batch["bundle_id"],
                batch["visit_mask"], batch["day_ts"], batch["agent_idx"],
                batch["dow"],
            )
            per_day = F.kl_div(
                F.log_softmax(logits, dim=-1), target, reduction="none"
            ).sum(-1)
            weights = valid.float()
            loss = (per_day * weights).sum() / weights.sum().clamp_min(1.0)

        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * n_days
        total_days += n_days

    return total_loss / total_days if total_days else float("nan")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stop: bool = True,
    patience: int = 50,
    log_every: int = 10,
) -> dict:
    """Fit the forecaster, keeping the best-validation weights.

    Scoring uses the best epoch rather than the last, so a model that starts
    overfitting late is not the one that gets evaluated. A cohort without
    validation days (NUMOSIM's protocol) keeps the last epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train": [], "val": []}
    best_val, best_state, stale = float("inf"), None, 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_kl = run_epoch(model, train_loader, device, optimizer, split_id=0)
        val_kl = run_epoch(model, val_loader, device, split_id=1)
        history["train"].append(train_kl)
        history["val"].append(val_kl)
        schedule.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  train={train_kl:.4f}  "
                  f"val={val_kl:.4f}  {time.time() - started:.0f}s")

        if val_kl != val_kl:                      # no validation days
            continue
        if val_kl < best_val:
            best_val, stale = val_kl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if early_stop and stale >= patience:
                print(f"    early stop at epoch {epoch} (best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val": best_val, "epochs_ran": len(history["train"])}
