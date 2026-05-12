"""Small physics-informed draft model for ART-style rendezvous controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DraftARTConfig:
    """Configuration for :class:`PhysicsInformedDraftART`."""

    state_dim: int = 6
    action_dim: int = 3
    context_dim: int = 4
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    draft_chunk_length: int = 8
    max_sequence_length: int = 128


class PhysicsInformedDraftART(nn.Module):
    """Causal transformer draft model for multi-step impulsive controls.

    Purpose:
        Predict a chunk ``draft_control_dv`` of RTN-frame impulsive controls from
        normalized ART history without predicting states by default.

    Inputs:
        state: ``float32[B, L + 1, 6]`` normalized ROE or RTN states. ROE is the
            preferred model input when available; RTN is used by repository ART
            checkpoints and the synthetic fallback. Position units are normalized
            model units; unnormalized RTN is ``[m, m/s]``.
        action: ``float32[B, L + 1, 3]`` normalized previous controls in ``[m/s]``
            after dataset normalization. The final/current action may be zero.
        fuel_to_go: ``float32[B, L + 1, 1]`` positive fuel-to-go ``phi_k`` in
            normalized ``[m/s]`` units.
        constraint_to_go: ``float32[B, L + 1, 1]`` normalized CTG ``psi_k``.
        time_features: optional ``float32[B, L + 1, 2]`` containing normalized
            elapsed time and normalized time-to-go. If omitted it is generated
            from sequence length.
        attention_mask: optional ``bool[B, L + 1]`` where True marks real tokens.
        horizon: optional requested output horizon ``H <= draft_chunk_length``.

    Outputs:
        ``float32[B, H, 3]`` draft RTN delta-v controls in normalized action
        units unless the caller denormalizes them with repository statistics.
    """

    def __init__(self, config: DraftARTConfig):
        super().__init__()
        self.config = config
        token_dim = config.state_dim + config.action_dim + 1 + 1 + 2
        self.input_projection = nn.Linear(token_dim, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embed_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.draft_chunk_length * config.action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        fuel_to_go: torch.Tensor,
        constraint_to_go: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = state.shape
        if time_features is None:
            elapsed = torch.linspace(0.0, 1.0, seq_len, device=state.device, dtype=state.dtype)
            elapsed = elapsed.view(1, seq_len, 1).expand(batch_size, -1, -1)
            time_features = torch.cat((elapsed, 1.0 - elapsed), dim=-1)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=state.device)
        tokens = torch.cat((state, action, fuel_to_go, constraint_to_go, time_features), dim=-1)
        positions = torch.arange(seq_len, device=state.device).clamp(max=self.config.max_sequence_length - 1)
        hidden = self.input_projection(tokens) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=state.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask.bool(),
        )
        last_indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = encoded[torch.arange(batch_size, device=state.device), last_indices]
        controls = self.output_head(pooled).view(
            batch_size, self.config.draft_chunk_length, self.config.action_dim
        )
        if horizon is None:
            horizon = self.config.draft_chunk_length
        return controls[:, :horizon, :]
