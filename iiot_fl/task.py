import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iiot_fl.model import IIoTFLNet, DualTaskLoss

logger = logging.getLogger(__name__)


def get_parameters(model: nn.Module) -> list:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list):
    from collections import OrderedDict

    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def train(
    model: IIoTFLNet,
    train_loader: DataLoader,
    criterion: DualTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    local_epochs: int,
) -> Dict[str, float]:
    model.train()
    final_metrics = {}

    for epoch in range(local_epochs):
        total_loss = 0.0
        total_rul = 0.0
        total_fail = 0.0
        n = 0

        for x, rul_true, fail_true in train_loader:
            x = x.to(device)
            rul_true = rul_true.to(device)
            fail_true = fail_true.to(device)

            optimizer.zero_grad()
            rul_pred, fail_logit = model(x)

            loss, rul_loss, fail_loss = criterion(
                rul_pred, rul_true, fail_logit, fail_true
            )
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = len(rul_true)
            total_loss += loss.item() * batch_n
            total_rul += rul_loss.item() * batch_n
            total_fail += fail_loss.item() * batch_n
            n += batch_n

        if scheduler is not None:
            scheduler.step()

        final_metrics = {
            "avg_loss": total_loss / n,
            "avg_rul_loss": total_rul / n,
            "avg_fail_loss": total_fail / n,
        }
        logger.info(
            "  Epoch %d/%d | loss=%.4f | rul=%.4f | fail=%.4f",
            epoch + 1,
            local_epochs,
            final_metrics["avg_loss"],
            final_metrics["avg_rul_loss"],
            final_metrics["avg_fail_loss"],
        )

    return final_metrics


def evaluate(
    model: IIoTFLNet,
    val_loader: DataLoader,
    criterion: DualTaskLoss,
    device: torch.device,
) -> Tuple[float, int, Dict[str, float]]:
    model.eval()

    total_loss = 0.0
    rul_mae = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0
    n = 0

    with torch.no_grad():
        for x, rul_true, fail_true in val_loader:
            x = x.to(device)
            rul_true = rul_true.to(device)
            fail_true = fail_true.to(device)

            rul_pred, fail_logit = model(x)
            loss, _, _ = criterion(rul_pred, rul_true, fail_logit, fail_true)

            rul_mae += torch.abs(rul_pred.squeeze() - rul_true).sum().item()

            preds = (torch.sigmoid(fail_logit.squeeze()) > 0.5).long()
            gt = fail_true.long()
            tp += ((preds == 1) & (gt == 1)).sum().item()
            fp += ((preds == 1) & (gt == 0)).sum().item()
            tn += ((preds == 0) & (gt == 0)).sum().item()
            fn += ((preds == 0) & (gt == 1)).sum().item()
            n += len(rul_true)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "rul_mae": rul_mae / n,
        "fail_accuracy": (tp + tn) / n,
        "fail_f1": f1,
        "fail_precision": precision,
        "fail_recall": recall,
    }

    logger.info(
        "  Eval | loss=%.4f | rul_mae=%.4f | fail_acc=%.4f | faill_f1=%.4f",
        total_loss / n,
        metrics["rul_mae"],
        metrics["fail_accuracy"],
        metrics["fail_f1"],
    )

    return float(total_loss / n), n, metrics
