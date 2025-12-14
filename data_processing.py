#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claims Data Processing Module
Handles all data preprocessing, feature engineering, and dataset creation for claims data.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# Feature column definitions
NUMERIC_RAW = [
    "ClaimAmount", "Reimbursement", "Clearinghouse_Edit_Count", 
    "Days_To_Appeal", "Benefit_Limit"
]

BIN_RAW = [
    "PriorAuth_Obtained", "Referral_Required"
]

CAT_RAW = [
    "PayerID", "ProviderID", "ICD10_Code", "CPT_Code", 
    "HCPCS_Code", "Modifier", "DRG_Code", "Plan_Type"
]

DATE_COLS = [
    "ServiceDate", "Contracted_Submission_Date", "SubmissionDate", 
    "ProcessedDate", "Coverage_Start_Date", "Coverage_End_Date"
]

TARGET_STATUS = "DenialStatus"
TARGET_CODE = "Denial_Code"
NONE_DENIAL = "None"


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns to datetime format."""
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


def _days_diff_pos(a: pd.Series, b: pd.Series) -> pd.Series:
    """Calculate positive day differences between two date series."""
    diff = (a - b).dt.days
    diff = diff.fillna(0).clip(lower=0)
    return diff


def engineer_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer causal features from raw data.
    
    Args:
        df: Raw dataframe with parsed dates
    
    Returns:
        DataFrame with engineered features added
    """
    # Timely filing lateness (days late after contracted)
    if "SubmissionDate" in df.columns and "Contracted_Submission_Date" in df.columns:
        df["timely_days_late"] = _days_diff_pos(
            df["SubmissionDate"], df["Contracted_Submission_Date"]
        )
    else:
        df["timely_days_late"] = 0

    # Over benefit amount (positive means over the limit)
    if "ClaimAmount" in df.columns and "Benefit_Limit" in df.columns:
        df["over_benefit_amt"] = (
            df["ClaimAmount"].fillna(0) - df["Benefit_Limit"].fillna(0)
        ).clip(lower=0)
    else:
        df["over_benefit_amt"] = 0

    # Coverage gap (positive if coverage ended before service)
    if "ServiceDate" in df.columns and "Coverage_End_Date" in df.columns:
        df["coverage_gap_days"] = _days_diff_pos(
            df["ServiceDate"], df["Coverage_End_Date"]
        )
    else:
        df["coverage_gap_days"] = 0

    # High edits indicator (binary)
    if "Clearinghouse_Edit_Count" in df.columns:
        df["high_edits"] = (
            df["Clearinghouse_Edit_Count"].fillna(0) >= 5
        ).astype(np.float32)
    else:
        df["high_edits"] = 0.0

    # Processing lag (submission -> processed), potentially leakage-prone
    if "ProcessedDate" in df.columns and "SubmissionDate" in df.columns:
        df["proc_lag_days"] = _days_diff_pos(
            df["ProcessedDate"], df["SubmissionDate"]
        )
    else:
        df["proc_lag_days"] = 0

    return df


def build_vocab(series: pd.Series, min_freq: int = 1) -> Dict[str, int]:
    """
    Build vocabulary for categorical features.
    
    Args:
        series: Pandas series with categorical values
        min_freq: Minimum frequency for inclusion in vocab
    
    Returns:
        Dictionary mapping values to indices
    """
    counts = series.astype(str).value_counts()
    vocab = {"<UNK>": 0}
    idx = 1
    for val, cnt in counts.items():
        if cnt >= min_freq:
            vocab[str(val)] = idx
            idx += 1
    return vocab


def map_to_ids(series: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
    """Map categorical values to vocabulary indices."""
    return series.astype(str).map(lambda x: vocab.get(x, 0)).astype(np.int64).values


def coerce_binary(df_bin: pd.DataFrame) -> pd.DataFrame:
    """
    Robust binary feature coercion handling bool, numeric, and string values.
    
    Args:
        df_bin: DataFrame with binary columns
    
    Returns:
        DataFrame with binary values as float32
    """
    mapping = {
        "y": 1.0, "yes": 1.0, "true": 1.0, "t": 1.0, "1": 1.0,
        "n": 0.0, "no": 0.0, "false": 0.0, "f": 0.0, "0": 0.0
    }
    out = {}
    
    for col in df_bin.columns:
        s = df_bin[col]
        
        # Handle booleans
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(np.float32)
            continue
        
        # Handle numeric dtypes: threshold at 0.5
        if pd.api.types.is_numeric_dtype(s):
            out[col] = (s.fillna(0).astype(np.float32) > 0.5).astype(np.float32)
            continue
        
        # Handle strings and objects: attempt numeric parse first
        down = s.astype(str).str.strip().str.lower()
        as_num = pd.to_numeric(down, errors="coerce")
        mask_num = as_num.notna()
        vec = pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        
        if mask_num.any():
            vec.loc[mask_num] = (as_num[mask_num] > 0.5).astype(np.float32).values
        
        # Map remaining tokens to boolean strings
        rem = ~mask_num
        if rem.any():
            mapped = down[rem].map(mapping)
            if mapped.isna().any():
                bad_vals = sorted(
                    v for v in mapped.index.map(down.__getitem__)[mapped.isna()].unique()
                )
                if bad_vals:
                    print(f"[WARN] Unexpected values in binary col '{col}': {bad_vals}")
            vec.loc[rem] = mapped.fillna(0.0).astype(np.float32).values
        
        out[col] = vec.values
    
    return pd.DataFrame(out, index=df_bin.index)


def precompute_env_weights(
    env_ids: np.ndarray, 
    clip: Tuple[float, float] = (0.5, 2.0)
) -> Dict[int, float]:
    """
    Compute inverse-frequency weighting by environment (payer).
    
    Args:
        env_ids: Array of environment IDs
        clip: Min and max values for weight clipping
    
    Returns:
        Dictionary mapping environment ID to weight
    """
    unique, counts = np.unique(env_ids, return_counts=True)
    inv = 1.0 / np.maximum(counts.astype(np.float32), 1.0)
    mean_inv = float(inv.mean()) if len(inv) else 1.0
    weights = inv / max(mean_inv, 1e-8)
    
    # Clip to avoid extreme scaling
    lo, hi = clip
    weights = np.clip(weights, lo, hi)
    
    return {int(u): float(w) for u, w in zip(unique.tolist(), weights.tolist())}


def get_feature_columns(guard_leakage: bool = False) -> Tuple[List[str], List[str]]:
    """
    Get the lists of numeric and binary feature columns.
    
    Args:
        guard_leakage: If True, remove potentially leaking features
    
    Returns:
        Tuple of (numeric_cols, binary_cols)
    """
    numeric_cols = NUMERIC_RAW + [
        "timely_days_late", "over_benefit_amt", "coverage_gap_days", "proc_lag_days"
    ]
    binary_cols = BIN_RAW + ["high_edits"]
    
    if guard_leakage:
        # Remove outcome-like fields for pre-submission prediction
        numeric_cols = [c for c in numeric_cols if c not in {"Reimbursement", "proc_lag_days"}]
    
    return numeric_cols, binary_cols


class ClaimsDataset(Dataset):
    """
    PyTorch Dataset for claims data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        vocabularies: Dict[str, Dict[str, int]],
        scaler: StandardScaler,
        numeric_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str],
        target_status: str,
        target_code: str,
        env_weight_map: Optional[Dict[int, float]] = None,
        denied_only_code: bool = True,
        ignore_index: int = -100,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.vocabularies = vocabularies
        self.scaler = scaler
        self.ignore_index = ignore_index
        self.denied_only_code = denied_only_code

        # Work on a local copy for transformations
        df_loc = df.copy()

        # Numeric: ensure float and impute NaN before scaling
        df_loc[numeric_cols] = df_loc[numeric_cols].astype(np.float32).fillna(0.0)
        self.X_num = scaler.transform(df_loc[numeric_cols].values)

        # Binary normalization
        self.X_bin = coerce_binary(df_loc[binary_cols]).values

        # Categorical
        self.X_cat = [
            map_to_ids(df_loc[c], vocabularies[c]) for c in categorical_cols
        ]

        # Targets
        self.y_status = df_loc[target_status].astype(np.int64).values

        # Denial code with denied-only semantics if requested
        code_vocab = vocabularies["_DENIAL_CODE_"]
        if self.denied_only_code:
            # Ignore non-denied claims for CE
            mapped = df_loc[target_code].astype(str).map(
                lambda x: code_vocab.get(x, 0)
            ).astype(np.int64).values
            self.y_code = np.where(
                df_loc[target_status].values == 1, 
                mapped, 
                self.ignore_index
            ).astype(np.int64)
        else:
            self.y_code = df_loc[target_code].astype(str).map(
                lambda x: code_vocab.get(x, 0)
            ).astype(np.int64).values

        # Environment (payer)
        payer_vocab = vocabularies["PayerID"]
        self.env = map_to_ids(df_loc["PayerID"], payer_vocab)

        # Stable per-sample environment weights
        if env_weight_map is not None:
            self.env_w = np.array(
                [env_weight_map.get(int(e), 1.0) for e in self.env], 
                dtype=np.float32
            )
        else:
            self.env_w = np.ones_like(self.env, dtype=np.float32)

    def __len__(self):
        return len(self.y_status)

    def __getitem__(self, idx: int):
        return {
            "num": torch.from_numpy(self.X_num[idx]).to(torch.float32),
            "bin": torch.from_numpy(self.X_bin[idx]).to(torch.float32),
            "cat": [torch.tensor(x[idx], dtype=torch.long) for x in self.X_cat],
            "y_status": torch.tensor(self.y_status[idx], dtype=torch.long),
            "y_code": torch.tensor(self.y_code[idx], dtype=torch.long),
            "env": torch.tensor(self.env[idx], dtype=torch.long),
            "env_w": torch.tensor(self.env_w[idx], dtype=torch.float32),
        }


class DataProcessor:
    """
    Main data processing class that encapsulates all preprocessing logic.
    """
    
    def __init__(
        self, 
        min_freq: int = 1, 
        guard_leakage: bool = False,
        val_split: float = 0.2,
        time_split: bool = False,
        seed: int = 42
    ):
        self.min_freq = min_freq
        self.guard_leakage = guard_leakage
        self.val_split = val_split
        self.time_split = time_split
        self.seed = seed
        
        self.vocabularies = {}
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.binary_cols = []
        self.categorical_cols = CAT_RAW
        self.num_denial_classes = 0
        self.env_weight_map = None
        self.pos_weight = None
    
    def prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
        
        Returns:
            Tuple of (train_df, val_df)
        """
        # Load and preprocess
        df = pd.read_csv(csv_path)
        df = parse_dates(df)
        df = engineer_causal_features(df)
        
        # Validate targets
        if TARGET_CODE not in df.columns:
            raise ValueError("Denial_Code column missing from CSV.")
        if TARGET_STATUS not in df.columns:
            raise ValueError("DenialStatus column missing from CSV.")
        
        # Get feature columns
        self.numeric_cols, self.binary_cols = get_feature_columns(self.guard_leakage)
        
        # Ensure required columns exist
        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = 0.0
        for c in self.binary_cols:
            if c not in df.columns:
                df[c] = 0.0
        for c in self.categorical_cols:
            if c not in df.columns:
                df[c] = "<UNK>"
        
        # Split data
        train_df, val_df = self._split_data(df)
        
        # Build vocabularies and fit scaler on training data
        self._fit_preprocessing(train_df)
        
        return train_df, val_df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets."""
        if self.time_split and "SubmissionDate" in df.columns:
            df = df.sort_values("SubmissionDate").reset_index(drop=True)
            split_idx = int(len(df) * (1.0 - self.val_split))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
        else:
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, 
                test_size=self.val_split, 
                random_state=self.seed, 
                stratify=df[TARGET_STATUS]
            )
        return train_df, val_df
    
    def _fit_preprocessing(self, train_df: pd.DataFrame):
        """Fit preprocessing components on training data."""
        # Build categorical vocabularies
        for c in self.categorical_cols:
            self.vocabularies[c] = build_vocab(train_df[c], min_freq=self.min_freq)
        
        # Build denial code vocabulary on denied-only rows
        denied_train_codes = train_df.loc[train_df[TARGET_STATUS] == 1, TARGET_CODE]
        self.vocabularies["_DENIAL_CODE_"] = build_vocab(denied_train_codes, min_freq=1)
        self.num_denial_classes = len(self.vocabularies["_DENIAL_CODE_"])
        
        # Fit scaler on numeric features
        train_df[self.numeric_cols] = train_df[self.numeric_cols].astype(np.float32).fillna(0.0)
        self.scaler.fit(train_df[self.numeric_cols].values)
        
        # Compute environment weights
        payer_ids_train = train_df["PayerID"].astype(str).map(
            self.vocabularies["PayerID"]
        ).fillna(0).astype(int).values
        self.env_weight_map = precompute_env_weights(payer_ids_train, clip=(0.5, 2.0))
        
        # Compute class imbalance weight
        p = float(train_df[TARGET_STATUS].mean())
        self.pos_weight = (1.0 - p) / max(p, 1e-6)
        print(f"[INFO] Train positive rate={p:.4f}, pos_weight={self.pos_weight:.3f}")
    
    def create_dataset(
        self, 
        df: pd.DataFrame, 
        use_env_weights: bool = False
    ) -> ClaimsDataset:
        """
        Create a ClaimsDataset from a DataFrame.
        
        Args:
            df: DataFrame to create dataset from
            use_env_weights: Whether to use environment weights
        
        Returns:
            ClaimsDataset instance
        """
        env_map = self.env_weight_map if use_env_weights else None
        
        return ClaimsDataset(
            df=df,
            vocabularies=self.vocabularies,
            scaler=self.scaler,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            binary_cols=self.binary_cols,
            target_status=TARGET_STATUS,
            target_code=TARGET_CODE,
            env_weight_map=env_map,
            denied_only_code=True
        )
    
    def get_model_config_params(self) -> Dict:
        """Get parameters needed for model configuration."""
        from hrm_model import emb_dim_for_cardinality
        
        cat_vocab_sizes = [len(self.vocabularies[c]) for c in self.categorical_cols]
        cat_emb_dims = [emb_dim_for_cardinality(n) for n in cat_vocab_sizes]
        
        return {
            "numeric_dim": len(self.numeric_cols),
            "binary_dim": len(self.binary_cols),
            "cat_vocab_sizes": cat_vocab_sizes,
            "cat_emb_dims": cat_emb_dims,
            "num_denial_classes": self.num_denial_classes
        }