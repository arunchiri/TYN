#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training program for Claims Denial Prediction using HRM model.
This script orchestrates the data processing and model training.
"""

import os
import argparse
from dataclasses import asdict
from typing import Optional, Tuple

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
)
from tqdm.auto import tqdm

# Import custom modules
from hrm_model import TabularHRMConfig, ClaimsSpecificHRM
from data_processing import DataProcessor


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility."""
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
    """
    Pick the appropriate device and dtype for training.
    
    Args:
        use_amp: Whether to use automatic mixed precision
        prefer_bf16: Whether to prefer bfloat16 over float16
    
    Returns:
        Tuple of (device, amp_dtype)
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if use_amp and dev.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return dev, torch.bfloat16
        else:
            return dev, torch.float16
    return dev, None  # No AMP dtype on CPU


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    denial_w: float,
    code_w: float,
    pos_weight: Optional[torch.Tensor],
):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to run on
        amp_enabled: Whether AMP is enabled
        amp_dtype: AMP dtype if enabled
        scaler: GradScaler for AMP
        denial_w: Weight for denial status loss
        code_w: Weight for denial code loss
        pos_weight: Positive class weight for BCE
    
    Returns:
        Tuple of (total_loss, bce_loss, ce_loss)
    """
    model.train()
    total_loss, total_bce, total_ce = 0.0, 0.0, 0.0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        # Move to device
        num = batch["num"].to(device, non_blocking=True)
        binf = batch["bin"].to(device, non_blocking=True)
        cat = [c.to(device, non_blocking=True) for c in batch["cat"]]
        y_status = batch["y_status"].to(device, non_blocking=True)
        y_code = batch["y_code"].to(device, non_blocking=True)
        env_w = batch["env_w"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        def forward_losses():
            out = model({"num": num, "bin": binf, "cat": cat})
            bce = F.binary_cross_entropy_with_logits(
                out["denial_logit"], y_status.float(), 
                weight=env_w, pos_weight=pos_weight
            )
            ce = F.cross_entropy(out["code_logits"], y_code, ignore_index=-100)
            loss = denial_w * bce + code_w * ce
            return loss, bce.detach(), ce.detach()

        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, bce_val, ce_val = forward_losses()
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, bce_val, ce_val = forward_losses()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = y_status.size(0)
        total_loss += loss.item() * bs
        total_bce += bce_val.item() * bs
        total_ce += ce_val.item() * bs
        n += bs

        pbar.set_postfix(loss=total_loss / n, bce=total_bce / n, ce=total_ce / n)

    return total_loss / n, total_bce / n, total_ce / n


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    denial_w: float,
    code_w: float,
    pos_weight: Optional[torch.Tensor] = None,
):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The neural network model
        loader: DataLoader for validation data
        device: Device to run on
        denial_w: Weight for denial status loss
        code_w: Weight for denial code loss
        pos_weight: Positive class weight for BCE
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss, total_bce, total_ce = 0.0, 0.0, 0.0
    n = 0
    all_probs, all_labels = [], []
    all_code_logits, all_code_labels = [], []

    for batch in loader:
        num = batch["num"].to(device, non_blocking=True)
        binf = batch["bin"].to(device, non_blocking=True)
        cat = [c.to(device, non_blocking=True) for c in batch["cat"]]
        y_status = batch["y_status"].to(device, non_blocking=True)
        y_code = batch["y_code"].to(device, non_blocking=True)

        out = model({"num": num, "bin": binf, "cat": cat})

        # Unweighted env in validation, but keep pos_weight consistent
        if pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                out["denial_logit"], y_status.float(), pos_weight=pos_weight
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                out["denial_logit"], y_status.float()
            )
        ce = F.cross_entropy(out["code_logits"], y_code, ignore_index=-100)

        bs = y_status.size(0)
        total_loss += (denial_w * bce + code_w * ce).item() * bs
        total_bce += bce.item() * bs
        total_ce += ce.item() * bs
        n += bs

        probs = torch.sigmoid(out["denial_logit"]).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_status.cpu().numpy())

        all_code_logits.append(out["code_logits"].cpu().numpy())
        all_code_labels.append(y_code.cpu().numpy())

    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    code_logits = np.concatenate(all_code_logits) if all_code_logits else np.empty((0, 0))
    code_true = np.concatenate(all_code_labels) if all_code_labels else np.array([])

    # Status metrics
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    y_hat = (y_prob >= 0.5).astype(int) if y_prob.size else np.array([])
    f1 = f1_score(y_true, y_hat) if y_hat.size else float("nan")
    brier = brier_score_loss(y_true, y_prob) if y_prob.size else float("nan")

    # Code metrics on denied-only (ignore_index = -100)
    denied_mask = code_true != -100
    if denied_mask.any():
        logits_d = code_logits[denied_mask]
        true_d = code_true[denied_mask]
        code_pred = logits_d.argmax(axis=1)
        code_acc = accuracy_score(true_d, code_pred)
        
        # Top-5 accuracy
        topk = 5
        topk_idx = np.argpartition(-logits_d, kth=min(topk, logits_d.shape[1]-1), axis=1)[:, :topk]
        top5 = (topk_idx == true_d[:, None]).any(axis=1).mean()
        
        # Macro-F1
        code_f1 = f1_score(true_d, code_pred, average="macro")
    else:
        code_acc, top5, code_f1 = float("nan"), float("nan"), float("nan")

    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "ce": total_ce / n,
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "brier": brier,
        "code_acc_denied": code_acc,
        "code_top5_denied": top5,
        "code_f1_macro_denied": code_f1,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Titanic Survival Prediction Model")
    
    # Data arguments
    parser.add_argument("--csv", type=str, default="dataset/train.csv", help="Path to train.csv")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--time_split", action="store_true", help="Use time-based split on SubmissionDate")
    parser.add_argument("--min_freq", type=int, default=1, help="Min frequency for categorical vocab")
    parser.add_argument("--guard_leakage", action="store_true", help="Drop leakage-prone features")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--H_layers", type=int, default=2, help="Number of H-level layers")
    parser.add_argument("--L_layers", type=int, default=2, help="Number of L-level layers")
    parser.add_argument("--H_cycles", type=int, default=2, help="Number of H-level cycles")
    parser.add_argument("--L_cycles", type=int, default=2, help="Number of L-level cycles")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--expansion", type=float, default=2.0, help="MLP expansion factor")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--lr_scheduler", choices=["none", "cosine", "step"], default="cosine", help="Learning rate scheduler")
    parser.add_argument("--denial_w", type=float, default=1.0, help="Weight for denial status loss")
    parser.add_argument("--code_w", type=float, default=0.8, help="Weight for denial code loss")
    
    # System arguments
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on GPU")
    parser.add_argument("--prefer_bf16", action="store_true", help="Prefer bfloat16 for AMP")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = DataProcessor(
        min_freq=args.min_freq,
        guard_leakage=args.guard_leakage,
        val_split=args.val_split,
        time_split=args.time_split,
        seed=args.seed
    )
    
    # Load and prepare data
    print("Loading and preparing data...")
    train_df, val_df = data_processor.prepare_data(args.csv)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_ds = data_processor.create_dataset(train_df, use_env_weights=True)
    val_ds = data_processor.create_dataset(val_df, use_env_weights=False)
    
    # Setup device and precision
    dev, amp_dtype = pick_device_and_dtype(use_amp=args.amp, prefer_bf16=args.prefer_bf16)
    print(f"Using device: {dev}, AMP dtype: {amp_dtype}")
    
    # Create data loaders
    pin_mem = args.pin_memory and (dev.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    
    # Initialize model
    print("Initializing model...")
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
    )
    
    model = ClaimsSpecificHRM(cfg, model_params["num_denial_classes"])
    model.to(dev)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup AMP
    amp_enabled = (args.amp and dev.type == "cuda" and amp_dtype is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.5)
    else:
        scheduler = None
    
    # Prepare positive weight tensor
    pos_weight_tensor = torch.tensor(data_processor.pos_weight, dtype=torch.float32).to(dev)
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience, bad_epochs = 5, 0
    ckpt_path = os.path.join(args.save_dir, "claims_hrm_model.pt")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to: {ckpt_path}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        tr_loss, tr_bce, tr_ce = train_one_epoch(
            model, train_loader, optimizer, dev, 
            amp_enabled, amp_dtype, scaler,
            denial_w=args.denial_w, code_w=args.code_w, 
            pos_weight=pos_weight_tensor
        )
        
        # Evaluate
        metrics = evaluate(
            model, val_loader, dev, 
            denial_w=args.denial_w, code_w=args.code_w, 
            pos_weight=pos_weight_tensor
        )
        
        # Print metrics
        print(
            f"  Train: loss={tr_loss:.4f} | bce={tr_bce:.4f} | ce={tr_ce:.4f}\n"
            f"  Val:   loss={metrics['loss']:.4f} | bce={metrics['bce']:.4f} | "
            f"ce={metrics['ce']:.4f} | AUC={metrics['auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f} | "
            f"Brier={metrics['brier']:.4f}\n"
            f"         code_acc={metrics['code_acc_denied']:.4f} | "
            f"code_top5={metrics['code_top5_denied']:.4f} | "
            f"code_f1={metrics['code_f1_macro_denied']:.4f}\n"
            f"  LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            bad_epochs = 0
            
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "metrics": metrics,
                    "vocabularies": data_processor.vocabularies,
                    "preprocessing": {
                        "scaler_mean": data_processor.scaler.mean_.tolist(),
                        "scaler_scale": data_processor.scaler.scale_.tolist(),
                        "numeric_cols": data_processor.numeric_cols,
                        "binary_cols": data_processor.binary_cols,
                        "categorical_cols": data_processor.categorical_cols,
                        "num_denial_classes": data_processor.num_denial_classes,
                        "pos_weight": data_processor.pos_weight,
                    },
                    "args": vars(args),
                },
                ckpt_path
            )
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {ckpt_path}")
    print("="*50)


if __name__ == "__main__":
    main()