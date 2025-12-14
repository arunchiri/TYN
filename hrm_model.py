#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class HRMTabularBlock(nn.Module):
    def __init__(self, dim: int, expansion: float, dropout: float = 0.0):
        super().__init__()
        self.mlp = SwiGLU(dim, expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mlp(x))


class HRMTabularReasoning(nn.Module):
    def __init__(self, dim: int, layers: int, expansion: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([HRMTabularBlock(dim, expansion, dropout) for _ in range(layers)])

    def forward(self, z: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        z = z + injection
        for layer in self.layers:
            z = layer(z)
        return z


@dataclass
class TabularHRMConfig:
    # Inputs
    numeric_dim: int
    binary_dim: int
    cat_vocab_sizes: List[int]
    cat_emb_dims: List[int]

    # Backbone
    hidden_size: int = 128
    expansion: float = 2.0
    dropout: float = 0.05
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 2

    # Heads: name -> output dimension
    # Examples:
    #   binary: {"y": 1}
    #   multiclass K: {"y": K}
    #   multitask: {"y1": 1, "y2": 5}
    output_heads: Dict[str, int] = field(default_factory=lambda: {"y": 1})


class TabularHRM(nn.Module):
    """
    Generic tabular HRM. No task/domain-specific assumptions.
    Forward returns: Dict[str, Tensor] with each head's raw output.
    """
    def __init__(self, cfg: TabularHRMConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings
        self.cat_embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(cfg.cat_vocab_sizes, cfg.cat_emb_dims)
        ])
        cat_total_dim = sum(cfg.cat_emb_dims)

        # Input projection
        in_dim = cfg.numeric_dim + cfg.binary_dim + cat_total_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden_size)

        # Hierarchical reasoning
        self.H_level = HRMTabularReasoning(cfg.hidden_size, cfg.H_layers, cfg.expansion, cfg.dropout)
        self.L_level = HRMTabularReasoning(cfg.hidden_size, cfg.L_layers, cfg.expansion, cfg.dropout)

        # Heads
        self.heads = nn.ModuleDict({name: nn.Linear(cfg.hidden_size, dim) for name, dim in cfg.output_heads.items()})

        # Learned initial states
        self.H_init = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(cfg.hidden_size))

    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # cat: list of tensors [B] or [B,]
        if len(self.cat_embs) > 0:
            cat_vecs = [emb(batch["cat"][i]) for i, emb in enumerate(self.cat_embs)]
            cat_concat = torch.cat(cat_vecs, dim=-1)
        else:
            cat_concat = torch.zeros((batch["num"].shape[0], 0), device=batch["num"].device)

        x = torch.cat([batch["num"], batch["bin"], cat_concat], dim=-1)
        x = self.input_proj(x)

        z_H = self.H_init.unsqueeze(0).expand(x.size(0), -1)
        z_L = self.L_init.unsqueeze(0).expand(x.size(0), -1)

        for _ in range(self.cfg.H_cycles):
            for _ in range(self.cfg.L_cycles):
                z_L = self.L_level(z_L, z_H + x)
            z_H = self.H_level(z_H, z_L)

        return z_H

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = self.extract_features(batch)
        return {name: head(features) for name, head in self.heads.items()}


def emb_dim_for_cardinality(n: int) -> int:
    return int(min(64, max(4, round(1.6 * (n ** 0.56)))))