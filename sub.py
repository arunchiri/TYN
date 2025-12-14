import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data_processing import DataProcessor, parse_basic_missing, engineer_features_titanic
from hrm_model import TabularHRMConfig, TabularHRM

# Paths
TEST_CSV = "dataset/test.csv"
MODEL_CKPT = "outputs/tabular_hrm.pt"     # <- generic checkpoint from new train.py
OUTPUT_CSV = "dataset/submission.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_scaler_from_checkpoint(pre: dict) -> StandardScaler:
    """
    Rebuild a StandardScaler from saved mean/scale.
    sklearn's StandardScaler expects a few attributes to exist.
    """
    scaler = StandardScaler()
    mean = np.array(pre["scaler_mean"], dtype=np.float64)
    scale = np.array(pre["scaler_scale"], dtype=np.float64)

    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = mean.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def main():
    # Load checkpoint
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    pre = ckpt.get("preprocessing", {}) or {}

    vocabularies = pre.get("vocabularies", None)
    numeric_cols = pre.get("numeric_cols", None)
    binary_cols = pre.get("binary_cols", None)
    categorical_cols = pre.get("categorical_cols", None)

    if vocabularies is None or numeric_cols is None or binary_cols is None or categorical_cols is None:
        raise ValueError(
            "Checkpoint missing preprocessing info. "
            "Make sure you are using the checkpoint produced by the new generic train.py (outputs/tabular_hrm.pt)."
        )

    # Build a DataProcessor and restore preprocessing state
    dp = DataProcessor(
        min_freq=1,
        guard_leakage=False,
        val_split=0.0,
        time_split=False,
        seed=42,
        target_col="Survived",
        sample_weight_col=None,  # keep None for inference
    )
    dp.vocabularies = vocabularies
    dp.numeric_cols = numeric_cols
    dp.binary_cols = binary_cols
    dp.categorical_cols = categorical_cols

    if pre.get("scaler_mean") is None or pre.get("scaler_scale") is None:
        raise ValueError("Checkpoint missing scaler_mean/scaler_scale.")
    dp.scaler = restore_scaler_from_checkpoint(pre)

    # Rebuild model config
    model_params = dp.get_model_config_params()
    cfg_dict = ckpt["config"]

    # Ensure output head matches training ("y" for binary)
    output_heads = cfg_dict.get("output_heads", {"y": 1})

    cfg = TabularHRMConfig(
        numeric_dim=model_params["numeric_dim"],
        binary_dim=model_params["binary_dim"],
        cat_vocab_sizes=model_params["cat_vocab_sizes"],
        cat_emb_dims=model_params["cat_emb_dims"],
        hidden_size=cfg_dict.get("hidden_size", 96),
        expansion=cfg_dict.get("expansion", 2.0),
        dropout=cfg_dict.get("dropout", 0.25),
        H_layers=cfg_dict.get("H_layers", 1),
        L_layers=cfg_dict.get("L_layers", 1),
        H_cycles=cfg_dict.get("H_cycles", 1),
        L_cycles=cfg_dict.get("L_cycles", 1),
        output_heads=output_heads,
    )

    model = TabularHRM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Use tuned threshold if it was saved (you used --tune_threshold)
    best_thr = 0.5
    metrics = ckpt.get("metrics", {}) or {}
    if "best_threshold" in metrics and metrics["best_threshold"] is not None:
        best_thr = float(metrics["best_threshold"])
    print(f"[INFO] Using threshold: {best_thr:.3f}")

    # Load test data
    print("Loading test data...")
    raw_test = pd.read_csv(TEST_CSV)
    passenger_ids = raw_test["PassengerId"].copy()

    test_df = parse_basic_missing(raw_test)
    test_df = engineer_features_titanic(test_df)

    # Ensure required columns exist
    for c in dp.numeric_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    for c in dp.binary_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    for c in dp.categorical_cols:
        if c not in test_df.columns:
            test_df[c] = "<UNK>"

    # Dummy label column so dataset can be created (not used for predictions)
    if "Survived" not in test_df.columns:
        test_df["Survived"] = 0

    # Create dataset and loader
    test_ds = dp.create_dataset(test_df, use_env_weights=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Predict probabilities
    print("Generating predictions...")
    probs_all = []

    with torch.no_grad():
        for batch in test_loader:
            num = batch["num"].to(DEVICE)
            binf = batch["bin"].to(DEVICE)
            cat = [c.to(DEVICE) for c in batch["cat"]]

            out = model({"num": num, "bin": binf, "cat": cat})
            logit = out["y"].squeeze(-1)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs_all.append(prob)

    probs = np.concatenate(probs_all, axis=0)
    preds = (probs >= best_thr).astype(int)

    # Write submission
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds})
    submission.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Predicted survivors: {int(preds.sum())} / {len(preds)}")


if __name__ == "__main__":
    main()