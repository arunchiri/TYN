#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic training script for tabular tasks using a generic HRM backbone.

Supports:
- binary classification (default): BCEWithLogits + AUC/PR-AUC/F1/Brier
- multiclass classification: CrossEntropy + Acc/Macro-F1/Top-5
- regression: MSE + RMSE/MAE/R2

This script is domain-agnostic: no claims/denial-specific parameters or naming.
It assumes your DataProcessor/Dataset returns batches with:
  - "num": FloatTensor [B, numeric_dim]
  - "bin": FloatTensor [B, binary_dim]
  - "cat": list[LongTensor] each [B]  (or a LongTensor [B, n_cat])
  - labels in one of: "y", "y_status", "label", or args.target_col in the batch
Optionally:
  - "sample_weight" or "env_w": FloatTensor [B] for per-sample weighting
"""

import os
import argparse
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from tqdm.auto import tqdm

from hrm_model import TabularHRMConfig, TabularHRM
from data_processing import DataProcessor


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device_and_dtype(
    use_amp: bool = False,
    prefer_bf16: bool = True
) -> Tuple[torch.device, Optional[torch.dtype]]:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_amp and dev.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return dev, torch.bfloat16
        return dev, torch.float16
    return dev, None


def _get_inputs_from_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    num = batch["num"].to(device, non_blocking=True)
    binf = batch["bin"].to(device, non_blocking=True)

    cat = batch.get("cat", [])
    if isinstance(cat, torch.Tensor):
        # cat is [B, n_cat] -> convert to list of [B]
        cat_list = [cat[:, i].to(device, non_blocking=True) for i in range(cat.size(1))]
    else:
        # cat is list[tensor]
        cat_list = [c.to(device, non_blocking=True) for c in cat]

    return {"num": num, "bin": binf, "cat": cat_list}


def _get_labels_from_batch(
    batch: Dict[str, Any],
    device: torch.device,
    target_col: str
) -> torch.Tensor:
    # Try common keys
    for k in ("y", "y_status", "label", target_col):
        if k in batch:
            y = batch[k]
            return y.to(device, non_blocking=True)
    raise KeyError(
        f"Could not find labels in batch. Tried keys: 'y', 'y_status', 'label', '{target_col}'. "
        f"Batch keys: {list(batch.keys())}"
    )


def _get_sample_weight_from_batch(
    batch: Dict[str, Any],
    device: torch.device,
    batch_size: int
) -> Optional[torch.Tensor]:
    # Accept either key; if none, return None (unweighted)
    w = batch.get("sample_weight", None)
    if w is None:
        w = batch.get("env_w", None)
    if w is None:
        return None
    w = w.to(device, non_blocking=True).float()
    if w.ndim == 0:
        w = w.expand(batch_size)
    return w


def _binary_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Choose threshold that maximizes F1 on a given set.
    Returns: (best_threshold, best_f1)
    """
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_i = int(np.argmax(f1s))
    # precision_recall_curve returns thresholds length = len(precision)-1
    best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5
    return best_thr, float(f1s[best_i])


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    head: str,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    pos_weight: Optional[torch.Tensor],
    target_col: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, batch in enumerate(pbar, start=1):
        inputs = _get_inputs_from_batch(batch, device)
        y = _get_labels_from_batch(batch, device, target_col)

        optimizer.zero_grad(set_to_none=True)

        def forward_and_loss() -> torch.Tensor:
            out = model(inputs)
            pred = out[head]
            bs = y.size(0)

            sample_w = _get_sample_weight_from_batch(batch, device, bs)

            if task == "binary":
                logit = pred.squeeze(-1)
                # BCEWithLogits: weight is per-sample; pos_weight is global class weight
                loss_vec = F.binary_cross_entropy_with_logits(
                    logit,
                    y.float(),
                    reduction="none",
                    pos_weight=pos_weight
                )
                if sample_w is not None:
                    loss_vec = loss_vec * sample_w
                return loss_vec.mean()

            if task == "multiclass":
                logits = pred  # [B, C]
                # If you want sample weights for CE, apply manual reduction:
                loss_vec = F.cross_entropy(logits, y.long(), reduction="none")
                if sample_w is not None:
                    loss_vec = loss_vec * sample_w
                return loss_vec.mean()

            if task == "regression":
                # y expected float
                pred_val = pred.squeeze(-1).float()
                y_val = y.float()
                loss_vec = (pred_val - y_val) ** 2
                if sample_w is not None:
                    loss_vec = loss_vec * sample_w
                return loss_vec.mean()

            raise ValueError(f"Unknown task: {task}")

        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = forward_and_loss()
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = forward_and_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=total_loss / max(n, 1))

    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    head: str,
    pos_weight: Optional[torch.Tensor],
    target_col: str,
    tune_threshold: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    # Collect for metrics
    ys = []
    preds = []

    for batch in loader:
        inputs = _get_inputs_from_batch(batch, device)
        y = _get_labels_from_batch(batch, device, target_col)
        out = model(inputs)
        pred = out[head]

        bs = int(y.size(0))
        sample_w = _get_sample_weight_from_batch(batch, device, bs)

        if task == "binary":
            logit = pred.squeeze(-1)
            loss_vec = F.binary_cross_entropy_with_logits(
                logit,
                y.float(),
                reduction="none",
                pos_weight=pos_weight
            )
            if sample_w is not None:
                loss_vec = loss_vec * sample_w
            loss = loss_vec.mean()

            prob = torch.sigmoid(logit).detach().cpu().numpy()
            ys.append(y.detach().cpu().numpy())
            preds.append(prob)

        elif task == "multiclass":
            logits = pred
            loss_vec = F.cross_entropy(logits, y.long(), reduction="none")
            if sample_w is not None:
                loss_vec = loss_vec * sample_w
            loss = loss_vec.mean()

            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            ys.append(y.detach().cpu().numpy())
            preds.append(prob)

        elif task == "regression":
            pred_val = pred.squeeze(-1).float()
            y_val = y.float()
            loss_vec = (pred_val - y_val) ** 2
            if sample_w is not None:
                loss_vec = loss_vec * sample_w
            loss = loss_vec.mean()

            ys.append(y_val.detach().cpu().numpy())
            preds.append(pred_val.detach().cpu().numpy())

        else:
            raise ValueError(f"Unknown task: {task}")

        total_loss += float(loss.item()) * bs
        n += bs

    metrics: Dict[str, float] = {"loss": total_loss / max(n, 1)}

    if not ys:
        return metrics

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    if task == "binary":
        # Handle degenerate splits
        if len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_pred))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_pred))
        else:
            metrics["auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

        if tune_threshold:
            thr, best_f1 = _binary_threshold_f1(y_true, y_pred)
            metrics["best_threshold"] = float(thr)
            y_hat = (y_pred >= thr).astype(int)
            metrics["f1"] = float(best_f1)
        else:
            y_hat = (y_pred >= 0.5).astype(int)
            metrics["f1"] = float(f1_score(y_true, y_hat))

        metrics["accuracy"] = float(accuracy_score(y_true, y_hat))
        metrics["brier"] = float(brier_score_loss(y_true, y_pred))

    elif task == "multiclass":
        y_hat = y_pred.argmax(axis=1)
        metrics["accuracy"] = float(accuracy_score(y_true, y_hat))
        metrics["f1_macro"] = float(f1_score(y_true, y_hat, average="macro"))

        topk = min(5, y_pred.shape[1])
        topk_idx = np.argpartition(-y_pred, kth=topk - 1, axis=1)[:, :topk]
        metrics["top5_accuracy"] = float((topk_idx == y_true[:, None]).any(axis=1).mean())

    elif task == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic Tabular HRM Trainer")

    # Data
    parser.add_argument("--csv", type=str, default="dataset/train.csv", help="Path to CSV")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--time_split", action="store_true", help="Time-based split if supported by DataProcessor")
    parser.add_argument("--min_freq", type=int, default=1, help="Min frequency for categorical vocab")
    parser.add_argument("--guard_leakage", action="store_true", help="Drop leakage-prone features if supported")

    # Task
    parser.add_argument("--task", choices=["binary", "multiclass", "regression"], default="binary")
    parser.add_argument("--target_col", type=str, default="Survived", help="Target column name (used if batch contains it)")
    parser.add_argument("--head", type=str, default="y", help="Model head name")
    parser.add_argument("--num_classes", type=int, default=2, help="For multiclass: number of classes")
    parser.add_argument("--tune_threshold", action="store_true", help="For binary: pick threshold maximizing F1 on val")

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--H_layers", type=int, default=2)
    parser.add_argument("--L_layers", type=int, default=2)
    parser.add_argument("--H_cycles", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--expansion", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # System
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on GPU")
    parser.add_argument("--prefer_bf16", action="store_true", help="Prefer bfloat16 for AMP")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs")

    args = parser.parse_args()

    set_seed(args.seed)

    # Data
    print("Initializing data processor...")
    data_processor = DataProcessor(
        min_freq=args.min_freq,
        guard_leakage=args.guard_leakage,
        val_split=args.val_split,
        time_split=args.time_split,
        seed=args.seed,
    )

    print("Loading and preparing data...")
    train_df, val_df = data_processor.prepare_data(args.csv)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    print("Creating datasets...")
    train_ds = data_processor.create_dataset(train_df, use_env_weights=True)
    val_ds = data_processor.create_dataset(val_df, use_env_weights=False)

    dev, amp_dtype = pick_device_and_dtype(use_amp=args.amp, prefer_bf16=args.prefer_bf16)
    print(f"Using device: {dev}, AMP dtype: {amp_dtype}")

    pin_mem = args.pin_memory and (dev.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    # Model heads
    if args.task == "binary":
        output_heads = {args.head: 1}
    elif args.task == "multiclass":
        output_heads = {args.head: int(args.num_classes)}
    else:  # regression
        output_heads = {args.head: 1}

    model_params = data_processor.get_model_config_params()
    cfg = TabularHRMConfig(
        numeric_dim=model_params["numeric_dim"],
        binary_dim=model_params["binary_dim"],
        cat_vocab_sizes=model_params["cat_vocab_sizes"],
        cat_emb_dims=model_params["cat_emb_dims"],
        hidden_size=args.hidden_size,
        expansion=args.expansion,
        dropout=args.dropout,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        output_heads=output_heads,
    )

    print("Initializing model...")
    model = TabularHRM(cfg).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # AMP
    amp_enabled = (args.amp and dev.type == "cuda" and amp_dtype is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )
    elif args.lr_scheduler == "step":
        step_size = max(1, args.epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    else:
        scheduler = None

    # pos_weight for binary
    pos_weight_tensor: Optional[torch.Tensor] = None
    if args.task == "binary":
        pw = getattr(data_processor, "pos_weight", None)
        if pw is not None:
            pos_weight_tensor = torch.tensor(float(pw), dtype=torch.float32, device=dev)
        else:
            pos_weight_tensor = None

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "tabular_hrm.pt")
    best_val_loss = float("inf")
    bad_epochs = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoint path: {ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=dev,
            task=args.task,
            head=args.head,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            pos_weight=pos_weight_tensor,
            target_col=args.target_col,
            grad_clip=args.grad_clip,
        )

        val = evaluate(
            model=model,
            loader=val_loader,
            device=dev,
            task=args.task,
            head=args.head,
            pos_weight=pos_weight_tensor,
            target_col=args.target_col,
            tune_threshold=args.tune_threshold,
        )

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Train: loss={tr['loss']:.4f}")
        if args.task == "binary":
            extra = f"AUC={val.get('auc', float('nan')):.4f} | PR-AUC={val.get('pr_auc', float('nan')):.4f} | F1={val.get('f1', float('nan')):.4f} | Brier={val.get('brier', float('nan')):.4f}"
            if "best_threshold" in val:
                extra += f" | thr={val['best_threshold']:.3f}"
            print(f"  Val:   loss={val['loss']:.4f} | {extra}")
        elif args.task == "multiclass":
            print(
                f"  Val:   loss={val['loss']:.4f} | acc={val.get('accuracy', float('nan')):.4f} | "
                f"f1_macro={val.get('f1_macro', float('nan')):.4f} | top5={val.get('top5_accuracy', float('nan')):.4f}"
            )
        else:
            print(
                f"  Val:   loss={val['loss']:.4f} | rmse={val.get('rmse', float('nan')):.4f} | "
                f"mae={val.get('mae', float('nan')):.4f} | r2={val.get('r2', float('nan')):.4f}"
            )
        print(f"  LR: {lr:.6f}")

        if scheduler is not None:
            scheduler.step()

        # Save best
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "metrics": val,
                    "args": vars(args),
                    # Store preprocessing if DataProcessor exposes it (safe to omit if not present)
                    "preprocessing": {
                        "vocabularies": getattr(data_processor, "vocabularies", None),
                        "numeric_cols": getattr(data_processor, "numeric_cols", None),
                        "binary_cols": getattr(data_processor, "binary_cols", None),
                        "categorical_cols": getattr(data_processor, "categorical_cols", None),
                        "pos_weight": getattr(data_processor, "pos_weight", None),
                        "scaler_mean": getattr(getattr(data_processor, "scaler", None), "mean_", None).tolist()
                        if getattr(getattr(data_processor, "scaler", None), "mean_", None) is not None
                        else None,
                        "scaler_scale": getattr(getattr(data_processor, "scaler", None), "scale_", None).tolist()
                        if getattr(getattr(data_processor, "scaler", None), "scale_", None) is not None
                        else None,
                    },
                },
                ckpt_path,
            )
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {ckpt_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()