# =============================================================================
# Distilled Reward Wrapper: Fully Differentiable Sequence-Only Oracle
# =============================================================================
#
# Replaces ViennaRewardWrapper with a distilled student model that takes
# only sequence one-hot [B, L, 4] as input — no Vienna RNA API needed.
# This makes the entire reward pipeline differentiable, fixing the reward
# hacking problem caused by detached structural features.
#
# Gradient Flow:
#   Diffusion → Gumbel-softmax → soft one-hot [B, 197, 4]
#       → RNABiMamba(d_input=4) → log2FC reward [B]
#       ↑ ALL input channels carry gradients (no detach)
#
# The student was distilled from a 5-fold RNABiMamba ensemble (d_input=16)
# using soft labels on 196k sequences. It achieves Pearson 0.97 vs teacher
# and 0.81 vs ground truth log2FC.
# =============================================================================

import sys
from pathlib import Path

import torch
import torch.nn as nn

import drakes_paths as dp
RNABiMamba = dp.import_rnabimamba()


class DistilledRewardWrapper(nn.Module):
    """Fully differentiable reward wrapper using a distilled sequence-only oracle.

    Takes soft one-hot [B, L, 4] from Gumbel-softmax and produces log2FC
    predictions [B] with gradients flowing through all 4 input channels.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = ckpt['config']

        self.student = RNABiMamba(
            d_input=config['d_input'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
        )
        self.student.load_state_dict(ckpt['model_state_dict'])
        self.student.to(device)
        self.student.eval()
        for param in self.student.parameters():
            param.requires_grad = False

        self.config = config
        self.metrics = ckpt.get('metrics', {})

    def forward(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        """Compute reward from soft one-hot sequences.

        Args:
            soft_onehot: [B, L, 4] soft one-hot from Gumbel-softmax.

        Returns:
            [B] log2FC predictions (higher = more stable).
        """
        # No straight-through needed — the student takes soft probabilities
        # directly and all 4 channels carry gradients.
        return self.student(soft_onehot)
