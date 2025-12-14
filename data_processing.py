#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Module (Generic + Titanic feature engineering)

- No "claims", "denial", or multi-task label assumptions.
- Produces a single target label: key "y"
- Produces model inputs:
    "num": float32 tensor [B, numeric_dim]
    "bin": float32 tensor [B, binary_dim]
    "cat": list of int64 tensors, each [B]
- Optionally provides per-sample weights:
    "sample_weight": float32 tensor [B]  (defaults to 1.0)

This is compatible with the generic train.py you’re using.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# -----------------------
# Feature engineering utils
# -----------------------

def extract_title(name: str) -> str:
    """Extract title from a passenger name like 'Braund, Mr. Owen Harris' -> 'Mr'."""
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r" ([A-Za-z]+)\.", name)
    return m.group(1) if m else "Unknown"


def parse_basic_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic missing value handling:
    - Age: impute by Title median if possible, then global median
    - Embarked: mode
    - Fare: median by Pclass if available, then global median
    """
    df = df.copy()

    # Age
    if "Age" in df.columns:
        if "Name" in df.columns and "Title" not in df.columns:
            df["Title"] = df["Name"].apply(extract_title)

        if "Title" in df.columns:
            title_mapping = {
                "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
                "Lady": "Mrs", "Sir": "Mr", "Don": "Mr",
                "Dona": "Mrs", "Countess": "Mrs", "Jonkheer": "Mr",
                "Col": "Mr", "Major": "Mr", "Capt": "Mr",
                "Dr": "Mr", "Rev": "Mr",
            }
            df["Title"] = df["Title"].replace(title_mapping)

            # fill Age by Title median
            for t in df["Title"].dropna().unique():
                mask = (df["Title"] == t) & (df["Age"].isna())
                if mask.any():
                    med = df.loc[df["Title"] == t, "Age"].median()
                    df.loc[mask, "Age"] = med

        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked
    if "Embarked" in df.columns:
        mode = df["Embarked"].mode()
        df["Embarked"] = df["Embarked"].fillna(mode.iloc[0] if len(mode) else "S")

    # Fare
    if "Fare" in df.columns:
        if "Pclass" in df.columns:
            for pc in df["Pclass"].dropna().unique():
                mask = (df["Pclass"] == pc) & (df["Fare"].isna())
                if mask.any():
                    med = df.loc[df["Pclass"] == pc, "Fare"].median()
                    df.loc[mask, "Fare"] = med
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df


def engineer_features_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Titanic-oriented feature engineering.
    Safe to run even if some raw columns are missing (it checks existence).
    """
    df = df.copy()

    # Title
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
        title_mapping = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Mrs", "Sir": "Mr", "Don": "Mr",
            "Dona": "Mrs", "Countess": "Mrs", "Jonkheer": "Mr",
            "Col": "Mr", "Major": "Mr", "Capt": "Mr",
            "Dr": "Mr", "Rev": "Mr",
        }
        df["Title"] = df["Title"].replace(title_mapping)

    # Family features
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(float)
        df["FamilySize_Cat"] = pd.cut(
            df["FamilySize"],
            bins=[0, 1, 4, 20],
            labels=["Solo", "Small", "Large"],
        ).astype(str)
    else:
        df["FamilySize"] = 1.0
        df["IsAlone"] = 1.0
        df["FamilySize_Cat"] = "Solo"

    # Age features
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teen", "Adult", "Middle", "Elderly"],
            include_lowest=True,
        ).astype(str)
        df["Age_Squared"] = df["Age"] ** 2
        df["Age_Cubed"] = df["Age"] ** 3
    else:
        df["Age"] = 0.0
        df["AgeGroup"] = "Unknown"
        df["Age_Squared"] = 0.0
        df["Age_Cubed"] = 0.0

    # Fare features
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["Fare_Log"] = np.log1p(df["Fare"].clip(lower=0))
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"].clip(lower=1)
        df["FarePerPerson_Log"] = np.log1p(df["FarePerPerson"].clip(lower=0))

        # quantile buckets (robust to duplicates)
        try:
            df["FareGroup"] = pd.qcut(
                df["Fare"],
                q=4,
                labels=["Low", "Medium", "High", "VeryHigh"],
                duplicates="drop",
            ).astype(str)
        except Exception:
            df["FareGroup"] = "Unknown"
    else:
        df["Fare"] = 0.0
        df["Fare_Log"] = 0.0
        df["FarePerPerson"] = 0.0
        df["FarePerPerson_Log"] = 0.0
        df["FareGroup"] = "Unknown"

    # Cabin features
    if "Cabin" in df.columns:
        df["HasCabin"] = (~df["Cabin"].isna()).astype(float)
        df["CabinDeck"] = df["Cabin"].fillna("U").astype(str).str[0]
        df["NumCabins"] = df["Cabin"].fillna("").apply(lambda x: len(str(x).split()) if str(x).strip() else 0)
    else:
        df["HasCabin"] = 0.0
        df["CabinDeck"] = "U"
        df["NumCabins"] = 0.0

    # Ticket features
    if "Ticket" in df.columns:
        # letters-only prefix
        df["TicketPrefix"] = df["Ticket"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).fillna("None")
        df["TicketLen"] = df["Ticket"].astype(str).str.len()
    else:
        df["TicketPrefix"] = "None"
        df["TicketLen"] = 0.0

    # Interaction features
    if "Sex" in df.columns and "Pclass" in df.columns:
        df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    else:
        df["Sex_Pclass"] = "Unknown"

    if "Age" in df.columns and "Pclass" in df.columns:
        df["Age_Pclass"] = df["Age"].astype(float) * df["Pclass"].astype(float)
    else:
        df["Age_Pclass"] = 0.0

    if "Sex" in df.columns and "AgeGroup" in df.columns:
        df["Sex_Age"] = df["Sex"].astype(str) + "_" + df["AgeGroup"].astype(str)
    else:
        df["Sex_Age"] = "Unknown"

    # Mother feature
    if all(c in df.columns for c in ["Sex", "Parch", "Age", "Title"]):
        df["IsMother"] = (
            (df["Sex"] == "female")
            & (df["Parch"].fillna(0) > 0)
            & (df["Age"].fillna(0) > 18)
            & (df["Title"].astype(str) != "Miss")
        ).astype(float)
    else:
        df["IsMother"] = 0.0

    # Ensure Embarked exists as categorical
    if "Embarked" not in df.columns:
        df["Embarked"] = "Unknown"

    return df


# -----------------------
# Encoding helpers
# -----------------------

def build_vocab(series: pd.Series, min_freq: int = 1) -> Dict[str, int]:
    counts = series.astype(str).value_counts()
    vocab: Dict[str, int] = {"<UNK>": 0}
    idx = 1
    for val, cnt in counts.items():
        if int(cnt) >= int(min_freq):
            vocab[str(val)] = idx
            idx += 1
    return vocab


def map_to_ids(series: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
    return series.astype(str).map(lambda x: vocab.get(str(x), 0)).astype(np.int64).values


def coerce_binary(df_bin: pd.DataFrame) -> pd.DataFrame:
    """
    Convert various representations into 0/1 float32.
    Numeric: > 0.5 -> 1 else 0
    """
    mapping = {
        "y": 1.0, "yes": 1.0, "true": 1.0, "t": 1.0, "1": 1.0,
        "n": 0.0, "no": 0.0, "false": 0.0, "f": 0.0, "0": 0.0,
    }
    out: Dict[str, np.ndarray] = {}

    for col in df_bin.columns:
        s = df_bin[col]

        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(np.float32).values
            continue

        if pd.api.types.is_numeric_dtype(s):
            out[col] = (s.fillna(0).astype(np.float32) > 0.5).astype(np.float32).values
            continue

        down = s.astype(str).str.strip().str.lower()
        as_num = pd.to_numeric(down, errors="coerce")
        mask_num = as_num.notna()

        vec = np.zeros(len(s), dtype=np.float32)
        if mask_num.any():
            vec[mask_num.values] = (as_num[mask_num] > 0.5).astype(np.float32).values

        rem = ~mask_num
        if rem.any():
            mapped = down[rem].map(mapping).fillna(0.0).astype(np.float32).values
            vec[rem.values] = mapped

        out[col] = vec

    return pd.DataFrame(out, index=df_bin.index)


def precompute_group_weights(
    group_ids: np.ndarray,
    clip: Tuple[float, float] = (0.5, 2.0)
) -> Dict[int, float]:
    """
    Inverse-frequency weights for a discrete group id array.
    Returns a dict: id -> weight
    """
    unique, counts = np.unique(group_ids, return_counts=True)
    inv = 1.0 / np.maximum(counts.astype(np.float32), 1.0)
    inv_mean = float(inv.mean()) if len(inv) else 1.0
    weights = inv / max(inv_mean, 1e-8)

    lo, hi = clip
    weights = np.clip(weights, lo, hi)

    return {int(u): float(w) for u, w in zip(unique.tolist(), weights.tolist())}


# -----------------------
# Feature sets
# -----------------------

# Numeric features (engineered + raw-like)
NUMERIC_COLS: List[str] = [
    "Age", "Fare", "SibSp", "Parch",
    "FamilySize", "FarePerPerson",
    "Age_Squared", "Age_Cubed",
    "Fare_Log", "FarePerPerson_Log",
    "NumCabins", "TicketLen", "Age_Pclass",
]

# Binary features (true binary only!)
# NOTE: Pclass MUST NOT be here.
BINARY_COLS: List[str] = [
    "IsAlone", "HasCabin", "IsMother",
]

# Categorical features (includes Pclass)
CATEGORICAL_COLS: List[str] = [
    "Pclass", "Sex", "Embarked", "AgeGroup", "Title",
    "FamilySize_Cat", "FareGroup", "CabinDeck",
    "TicketPrefix", "Sex_Pclass", "Sex_Age",
]


# -----------------------
# Dataset
# -----------------------

class TabularDataset(Dataset):
    """
    Generic dataset yielding:
      - num, bin, cat
      - y
      - sample_weight
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocabularies: Dict[str, Dict[str, int]],
        scaler: StandardScaler,
        numeric_cols: List[str],
        binary_cols: List[str],
        categorical_cols: List[str],
        target_col: str,
        sample_weight_map: Optional[Dict[int, float]] = None,
        sample_weight_col: Optional[str] = None,
    ):
        self.numeric_cols = numeric_cols
        self.binary_cols = binary_cols
        self.categorical_cols = categorical_cols
        self.vocabularies = vocabularies
        self.scaler = scaler
        self.target_col = target_col

        df_loc = df.copy()

        # Ensure required cols exist
        for c in numeric_cols:
            if c not in df_loc.columns:
                df_loc[c] = 0.0
        for c in binary_cols:
            if c not in df_loc.columns:
                df_loc[c] = 0.0
        for c in categorical_cols:
            if c not in df_loc.columns:
                df_loc[c] = "<UNK>"

        # Numeric
        df_loc[numeric_cols] = df_loc[numeric_cols].astype(np.float32).fillna(0.0)
        self.X_num = scaler.transform(df_loc[numeric_cols].values).astype(np.float32)

        # Binary
        self.X_bin = coerce_binary(df_loc[binary_cols]).values.astype(np.float32)

        # Categorical (list of int arrays)
        self.X_cat = [map_to_ids(df_loc[c], vocabularies[c]) for c in categorical_cols]

        # Target
        if target_col not in df_loc.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        self.y = df_loc[target_col].astype(np.int64).values

        # Sample weights (optional)
        n = len(df_loc)
        weights = np.ones(n, dtype=np.float32)
        if sample_weight_map is not None and sample_weight_col is not None and sample_weight_col in df_loc.columns:
            # map group to vocab id space if it is categorical, else treat as raw int ids
            if sample_weight_col in vocabularies:
                ids = map_to_ids(df_loc[sample_weight_col], vocabularies[sample_weight_col])
            else:
                ids = pd.to_numeric(df_loc[sample_weight_col], errors="coerce").fillna(0).astype(int).values
            weights = np.array([sample_weight_map.get(int(i), 1.0) for i in ids], dtype=np.float32)
        self.sample_weight = weights

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "num": torch.from_numpy(self.X_num[idx]),
            "bin": torch.from_numpy(self.X_bin[idx]),
            "cat": [torch.tensor(col[idx], dtype=torch.long) for col in self.X_cat],
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "sample_weight": torch.tensor(self.sample_weight[idx], dtype=torch.float32),
        }


# -----------------------
# DataProcessor
# -----------------------

class DataProcessor:
    """
    Generic processor with Titanic feature engineering.
    """

    def __init__(
        self,
        min_freq: int = 1,
        guard_leakage: bool = False,
        val_split: float = 0.2,
        time_split: bool = False,
        seed: int = 42,
        target_col: str = "Survived",
        sample_weight_col: Optional[str] = None,  # e.g., "Sex" if you really want reweighting
    ):
        self.min_freq = int(min_freq)
        self.guard_leakage = bool(guard_leakage)
        self.val_split = float(val_split)
        self.time_split = bool(time_split)
        self.seed = int(seed)

        self.target_col = target_col
        self.sample_weight_col = sample_weight_col

        self.vocabularies: Dict[str, Dict[str, int]] = {}
        self.scaler = StandardScaler()

        self.numeric_cols: List[str] = list(NUMERIC_COLS)
        self.binary_cols: List[str] = list(BINARY_COLS)
        self.categorical_cols: List[str] = list(CATEGORICAL_COLS)

        self.pos_weight: Optional[float] = None
        self.sample_weight_map: Optional[Dict[int, float]] = None

    def prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(csv_path)

        # Basic cleanup + FE
        df = parse_basic_missing(df)
        df = engineer_features_titanic(df)

        # Validate target
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from CSV.")

        # Optional leakage guarding (Titanic: PassengerId is typically safe to drop)
        if self.guard_leakage:
            for col in ["PassengerId"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

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

        train_df, val_df = self._split_data(df)
        self._fit_preprocessing(train_df)

        return train_df, val_df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.time_split and "SubmissionDate" in df.columns:
            df = df.sort_values("SubmissionDate").reset_index(drop=True)
            split_idx = int(len(df) * (1.0 - self.val_split))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
            return train_df, val_df

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_split,
            random_state=self.seed,
            stratify=df[self.target_col],
        )
        return train_df, val_df

    def _fit_preprocessing(self, train_df: pd.DataFrame) -> None:
        # Vocabularies for categoricals
        for c in self.categorical_cols:
            self.vocabularies[c] = build_vocab(train_df[c], min_freq=self.min_freq)

        # Fit scaler on numeric
        train_num = train_df[self.numeric_cols].astype(np.float32).fillna(0.0).values
        self.scaler.fit(train_num)

        # Compute pos_weight for binary classification (used by train.py if present)
        p = float(train_df[self.target_col].mean())
        self.pos_weight = (1.0 - p) / max(p, 1e-6)
        print(f"[INFO] Train positive rate={p:.4f}, pos_weight={self.pos_weight:.3f}")

        # Optional group-based sample weights
        self.sample_weight_map = None
        if self.sample_weight_col is not None and self.sample_weight_col in train_df.columns:
            # Build vocab for sample_weight_col if it is categorical and not already built
            if self.sample_weight_col in self.categorical_cols and self.sample_weight_col not in self.vocabularies:
                self.vocabularies[self.sample_weight_col] = build_vocab(train_df[self.sample_weight_col], min_freq=1)

            if self.sample_weight_col in self.vocabularies:
                ids = train_df[self.sample_weight_col].astype(str).map(self.vocabularies[self.sample_weight_col]).fillna(0).astype(int).values
            else:
                ids = pd.to_numeric(train_df[self.sample_weight_col], errors="coerce").fillna(0).astype(int).values

            self.sample_weight_map = precompute_group_weights(ids, clip=(0.5, 2.0))

    def create_dataset(self, df: pd.DataFrame, use_env_weights: bool = False) -> TabularDataset:
        # Generic trainer will use sample_weight if present; otherwise harmless.
        wmap = self.sample_weight_map if (use_env_weights and self.sample_weight_map is not None) else None

        return TabularDataset(
            df=df,
            vocabularies=self.vocabularies,
            scaler=self.scaler,
            numeric_cols=self.numeric_cols,
            binary_cols=self.binary_cols,
            categorical_cols=self.categorical_cols,
            target_col=self.target_col,
            sample_weight_map=wmap,
            sample_weight_col=self.sample_weight_col,
        )

    def get_model_config_params(self) -> Dict[str, object]:
        from hrm_model import emb_dim_for_cardinality

        cat_vocab_sizes = [len(self.vocabularies[c]) for c in self.categorical_cols]
        cat_emb_dims = [emb_dim_for_cardinality(n) for n in cat_vocab_sizes]

        return {
            "numeric_dim": len(self.numeric_cols),
            "binary_dim": len(self.binary_cols),
            "cat_vocab_sizes": cat_vocab_sizes,
            "cat_emb_dims": cat_emb_dims,
        }