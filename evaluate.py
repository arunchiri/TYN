#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script to check model predictions and accuracy on test data.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from torch.utils.data import DataLoader

from data_processing import DataProcessor
from hrm_model import TabularHRMConfig, ClaimsSpecificHRM


def main():
    """Evaluate the trained model."""
    
    # Load data
    print("Loading data...")
    data_processor = DataProcessor(
        min_freq=1,
        guard_leakage=False,
        val_split=0.2,
        time_split=False,
        seed=42
    )
    
    train_df, val_df = data_processor.prepare_data("dataset/train.csv")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_ds = data_processor.create_dataset(train_df, use_env_weights=False)
    val_ds = data_processor.create_dataset(val_df, use_env_weights=False)
    
    # Create data loaders
    device = torch.device("cpu")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # Load model
    print("Loading model...")
    ckpt_path = "outputs/claims_hrm_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Model checkpoint not found at {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    config_dict = checkpoint["config"]
    model_params = data_processor.get_model_config_params()
    
    cfg = TabularHRMConfig(
        numeric_dim=model_params["numeric_dim"],
        binary_dim=model_params["binary_dim"],
        cat_vocab_sizes=model_params["cat_vocab_sizes"],
        cat_emb_dims=model_params["cat_emb_dims"],
        hidden_size=config_dict.get("hidden_size", 128),
        expansion=config_dict.get("expansion", 2.0),
        dropout=config_dict.get("dropout", 0.05),
        H_layers=config_dict.get("H_layers", 2),
        L_layers=config_dict.get("L_layers", 2),
        H_cycles=config_dict.get("H_cycles", 2),
        L_cycles=config_dict.get("L_cycles", 2),
    )
    
    model = ClaimsSpecificHRM(cfg, model_params["num_denial_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print("=" * 70)
    print("TRAINING SET EVALUATION")
    print("=" * 70)
    eval_loader(model, train_loader, device, "Training")
    
    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)
    eval_loader(model, val_loader, device, "Validation")


def eval_loader(model, loader, device, set_name):
    """Evaluate model on a data loader."""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            num = batch["num"].to(device)
            binf = batch["bin"].to(device)
            cat = [c.to(device) for c in batch["cat"]]
            y_status = batch["y_status"].to(device)
            
            out = model({"num": num, "bin": binf, "cat": cat})
            
            # Get predictions
            probs = torch.sigmoid(out["denial_logit"]).cpu().numpy()
            preds = (probs >= 0.5).astype(int).flatten()
            labels = y_status.cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs.flatten())
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float("nan")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Survival rate
    survival_rate_true = np.mean(all_labels)
    survival_rate_pred = np.mean(all_preds)
    
    # Print results
    print(f"\n{set_name} Set Results:")
    print(f"  Total Samples: {len(all_labels)}")
    print(f"\n  Metrics:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives (TN):  {tn:5d}  (Correctly predicted Did Not Survive)")
    print(f"    False Positives (FP): {fp:5d}  (Incorrectly predicted Survived)")
    print(f"    False Negatives (FN): {fn:5d}  (Incorrectly predicted Did Not Survive)")
    print(f"    True Positives (TP):  {tp:5d}  (Correctly predicted Survived)")
    
    print(f"\n  Survival Rate:")
    print(f"    True Survival Rate:      {survival_rate_true:.4f} ({int(np.sum(all_labels))} / {len(all_labels)})")
    print(f"    Predicted Survival Rate: {survival_rate_pred:.4f} ({int(np.sum(all_preds))} / {len(all_preds)})")
    
    print(f"\n  Correct Predictions: {tp + tn} / {len(all_labels)}")
    print(f"  Wrong Predictions:   {fp + fn} / {len(all_labels)}")
    
    print(f"\n  Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=["Did Not Survive", "Survived"]))


if __name__ == "__main__":
    main()
