#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Purpose Hierarchical Reasoning Machine (HRM) Model for Tabular Data
This module provides a reusable HRM architecture that can be applied to various tabular tasks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU activation function with gating mechanism."""
    
    def __init__(self, dim: int, expansion: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))
        return x


class HRMTabularBlock(nn.Module):
    """Basic building block for HRM with residual connection and layer norm."""
    
    def __init__(self, dim: int, expansion: float, dropout: float = 0.0):
        super().__init__()
        self.mlp = SwiGLU(dim, expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x + self.mlp(x))
        return x


class HRMTabularReasoning(nn.Module):
    """Hierarchical reasoning module with multiple layers."""
    
    def __init__(self, dim: int, layers: int, expansion: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMTabularBlock(dim, expansion, dropout) for _ in range(layers)
        ])

    def forward(self, z: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        z = z + injection
        for layer in self.layers:
            z = layer(z)
        return z


@dataclass
class TabularHRMConfig:
    """Configuration for the general purpose tabular HRM model."""
    
    # Input dimensions
    numeric_dim: int
    binary_dim: int
    cat_vocab_sizes: List[int]
    cat_emb_dims: List[int]
    
    # Model architecture
    hidden_size: int = 128
    expansion: float = 2.0
    dropout: float = 0.05
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 2
    
    # Task-specific output dimensions
    output_heads: Dict[str, int] = None  # e.g., {"classification": 2, "regression": 1}
    
    def __post_init__(self):
        if self.output_heads is None:
            self.output_heads = {"default": 1}


class GeneralTabularHRM(nn.Module):
    """
    General Purpose Hierarchical Reasoning Machine for Tabular Data.
    
    This model can be adapted for various tabular tasks including:
    - Binary/Multi-class classification
    - Multi-task learning
    - Regression
    - Ranking
    """
    
    def __init__(self, cfg: TabularHRMConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings (per-cardinality dims)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(cfg.cat_vocab_sizes, cfg.cat_emb_dims)
        ])
        cat_total_dim = sum(cfg.cat_emb_dims)

        # Input projection
        in_dim = cfg.numeric_dim + cfg.binary_dim + cat_total_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden_size)

        # Two-level HRM-like reasoning
        self.H_level = HRMTabularReasoning(
            cfg.hidden_size, cfg.H_layers, cfg.expansion, cfg.dropout
        )
        self.L_level = HRMTabularReasoning(
            cfg.hidden_size, cfg.L_layers, cfg.expansion, cfg.dropout
        )

        # Task-specific output heads
        self.output_heads = nn.ModuleDict({
            name: nn.Linear(cfg.hidden_size, dim) 
            for name, dim in cfg.output_heads.items()
        })

        # Learned initial states
        self.H_init = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(cfg.hidden_size))

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from raw inputs.
        
        Args:
            batch: Dictionary containing:
                - "num": numeric features [batch_size, numeric_dim]
                - "bin": binary features [batch_size, binary_dim]
                - "cat": list of categorical features
        
        Returns:
            Feature tensor [batch_size, hidden_size]
        """
        # Build tabular embedding
        if len(self.cat_embs):
            cat_vecs = [emb(batch["cat"][i]) for i, emb in enumerate(self.cat_embs)]
            cat_concat = torch.cat(cat_vecs, dim=-1)
        else:
            cat_concat = torch.zeros((batch["num"].shape[0], 0), device=batch["num"].device)

        x = torch.cat([batch["num"], batch["bin"], cat_concat], dim=-1)
        x = self.input_proj(x)

        # Expand initial states to batch
        z_H = self.H_init.unsqueeze(0).expand(x.size(0), -1)
        z_L = self.L_init.unsqueeze(0).expand(x.size(0), -1)

        # ACT-like cycles for hierarchical reasoning
        for _ in range(self.cfg.H_cycles):
            for _ in range(self.cfg.L_cycles):
                z_L = self.L_level(z_L, z_H + x)
            z_H = self.H_level(z_H, z_L)

        return z_H

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch dictionary
        
        Returns:
            Dictionary of output tensors for each head
        """
        features = self.extract_features(batch)
        
        outputs = {}
        for head_name, head_module in self.output_heads.items():
            outputs[head_name] = head_module(features)
        
        return outputs

    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the learned embeddings/representations for inputs.
        Useful for transfer learning or feature extraction.
        
        Args:
            batch: Input batch dictionary
        
        Returns:
            Feature embeddings [batch_size, hidden_size]
        """
        return self.extract_features(batch)


class ClaimsSpecificHRM(GeneralTabularHRM):
    """
    Claims-specific extension of the general HRM model.
    Provides specialized output formatting for the claims denial task.
    """
    
    def __init__(self, cfg: TabularHRMConfig, num_denial_classes: int):
        # Set up specific output heads for claims task
        cfg.output_heads = {
            "denial_status": 1,  # Binary classification
            "denial_code": num_denial_classes  # Multi-class classification
        }
        super().__init__(cfg)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with claims-specific output formatting.
        
        Returns:
            Dictionary with:
                - "denial_logit": Binary classification logit
                - "code_logits": Multi-class classification logits
        """
        outputs = super().forward(batch)
        
        # Reshape and rename for compatibility
        return {
            "denial_logit": outputs["denial_status"].squeeze(-1),
            "code_logits": outputs["denial_code"]
        }


def emb_dim_for_cardinality(n: int) -> int:
    """
    Heuristic for determining embedding dimension based on cardinality.
    
    Args:
        n: Cardinality (number of unique values)
    
    Returns:
        Suggested embedding dimension
    """
    return int(min(64, max(4, round(1.6 * (n ** 0.56)))))