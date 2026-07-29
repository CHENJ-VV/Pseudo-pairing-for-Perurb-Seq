"""MLP architectures and user-accessible model factories.

Users can either instantiate these defaults or pass custom factory functions in
CONFIG.forward_model_init_fn / CONFIG.inverse_model_init_fn.  A factory receives
(n_genes, n_perturbations, config) and returns torch.nn.Module.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class ForwardPerturbationMLP(nn.Module):
    """Predict strategy-specific perturbation delta from pseudo-control x0 and perturbation ID."""

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        pert_emb_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pert_embedding = nn.Embedding(n_perturbations, pert_emb_dim)
        self.x_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + pert_emb_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x0: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.x_encoder(x0), self.pert_embedding(pert_id)], dim=1))


class InversePerturbationMLP(nn.Module):
    """Multiclass perturbation-combination classifier from an effect vector."""

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_perturbations),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class InverseMultiLabelGeneMLP(nn.Module):
    """Predict perturbed gene-set membership using sigmoid outputs over genes.

    This is useful for dual/multi perturbations when the target should be the
    component genes rather than a single combination-class ID.
    """

    def __init__(self, n_genes: int, n_target_genes: int, hidden_dim: int = 512, latent_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_target_genes),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


def _cfg(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def default_forward_model_init(n_genes: int, n_perturbations: int, config: Any) -> nn.Module:
    kwargs = dict(_cfg(config, "forward_model_kwargs", {}) or {})
    kwargs.setdefault("pert_emb_dim", _cfg(config, "pert_emb_dim", 128))
    kwargs.setdefault("hidden_dim", _cfg(config, "hidden_dim", 512))
    kwargs.setdefault("latent_dim", _cfg(config, "latent_dim", 256))
    kwargs.setdefault("dropout", _cfg(config, "dropout", 0.15))
    return ForwardPerturbationMLP(n_genes=n_genes, n_perturbations=n_perturbations, **kwargs)


def default_inverse_model_init(n_genes: int, n_perturbations: int, config: Any) -> nn.Module:
    kwargs = dict(_cfg(config, "inverse_model_kwargs", {}) or {})
    kwargs.setdefault("hidden_dim", _cfg(config, "hidden_dim", 512))
    kwargs.setdefault("latent_dim", _cfg(config, "latent_dim", 256))
    kwargs.setdefault("dropout", _cfg(config, "dropout", 0.15))
    return InversePerturbationMLP(n_genes=n_genes, n_perturbations=n_perturbations, **kwargs)


def build_forward_model(n_genes: int, n_perturbations: int, config: Any) -> nn.Module:
    factory = _cfg(config, "forward_model_init_fn", None)
    if factory is None:
        return default_forward_model_init(n_genes, n_perturbations, config)
    return factory(n_genes=n_genes, n_perturbations=n_perturbations, config=config)


def build_inverse_model(n_genes: int, n_perturbations: int, config: Any) -> nn.Module:
    factory = _cfg(config, "inverse_model_init_fn", None)
    if factory is None:
        return default_inverse_model_init(n_genes, n_perturbations, config)
    return factory(n_genes=n_genes, n_perturbations=n_perturbations, config=config)
