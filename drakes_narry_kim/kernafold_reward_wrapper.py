# =============================================================================
# KernaFold Reward Wrapper: Fully Differentiable 16-Channel Reward
# =============================================================================
#
# Uses KernaFold (differentiable Vienna RNA surrogate) to produce structure
# and scalar features, then feeds all 16 channels to the RNABiMamba teacher.
# Unlike ViennaRewardWrapper (12/16 channels detached) or DistilledRewardWrapper
# (4-channel only), ALL 16 channels carry gradients back to the diffusion model.
#
# Gradient Flow:
#   Diffusion → Gumbel-softmax → soft_onehot [B, 197, 4]
#       → straight-through → KernaFold → struct logits + scalars
#       → straight-through on structs → 16-channel assembly
#       → frozen RNABiMamba teacher → reward [B]
#       ↑ ALL 16 input channels carry gradients
# =============================================================================

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import KernaFold and RNABiMamba via importlib to avoid model.py name clash
import importlib.util

def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_kernafold_module = _import_from_path(
    'kernafold_model', '/mnt/ssd1/code/kernafold/model.py')
build_kernafold = _kernafold_module.build_model

_narry_kim_module = _import_from_path(
    'narry_kim_model', '/mnt/ssd1/code/narry_kim_2025/models/model.py')
RNABiMamba = _narry_kim_module.RNABiMamba

# Teacher scalar normalization divisors (matching ViennaRewardWrapper encoding)
# Order: mfe_energy, centroid_energy, centroid_distance,
#         ensemble_energy, mfe_frequency, ensemble_diversity
_SCALAR_DIVISORS = [100.0, 100.0, 50.0, 100.0, 1.0, 50.0]


class KernaFoldRewardWrapper(nn.Module):
    """Fully differentiable reward: KernaFold surrogate + RNABiMamba teacher.

    All 16 input channels to the teacher carry gradients back to the
    diffusion model, preventing the reward hacking observed with
    ViennaRewardWrapper (4/16 channels) and DistilledRewardWrapper (4/4
    channels but wrong landscape).

    Args:
        kernafold_checkpoint_path: Path to KernaFold .pt checkpoint.
        regressor_checkpoint_dir: Path to RNABiMamba k-fold checkpoint dir.
        device: Target device.
    """

    def __init__(self, kernafold_checkpoint_path: str,
                 regressor_checkpoint_dir: str, device: str = 'cuda'):
        super().__init__()

        # --- Load KernaFold (frozen, but forward is differentiable) ---
        ckpt = torch.load(kernafold_checkpoint_path, map_location=device,
                          weights_only=False)
        cfg = ckpt['config']
        self.kernafold = build_kernafold(
            variant=cfg['variant'],
            d_model=cfg['d_model'],
            n_conv_layers=cfg['n_conv_layers'],
            n_transformer_layers=cfg['n_transformer_layers'],
            n_heads=cfg['n_heads'],
            dropout=cfg['dropout'],
            scalar_mean=cfg.get('scalar_mean'),
            scalar_std=cfg.get('scalar_std'),
            d_pair=cfg.get('d_pair', 64),
            n_pair_layers=cfg.get('n_pair_layers', 8),
        )
        self.kernafold.load_state_dict(ckpt['model_state_dict'])
        self.kernafold.to(device)
        self.kernafold.eval()
        for p in self.kernafold.parameters():
            p.requires_grad = False
        self.kernafold_variant = cfg['variant']
        n_kf = sum(p.numel() for p in self.kernafold.parameters())
        print(f'  KernaFold ({cfg["variant"]}): {n_kf:,} params')

        # --- Load RNABiMamba teacher (frozen) ---
        run_dir = Path(regressor_checkpoint_dir)
        with open(run_dir / 'config.json') as f:
            reg_config = json.load(f)
        self.regressor = RNABiMamba(
            d_input=reg_config['d_input'],
            d_model=reg_config['d_model'],
            n_layers=reg_config['n_layers'],
            dropout=reg_config['dropout'],
        )
        reg_ckpt = torch.load(
            run_dir / 'fold_0' / 'best_model_sig.pt',
            map_location=device, weights_only=True)
        self.regressor.load_state_dict(reg_ckpt['model_state_dict'])
        self.regressor.to(device)
        self.regressor.eval()
        for p in self.regressor.parameters():
            p.requires_grad = False
        n_reg = sum(p.numel() for p in self.regressor.parameters())
        print(f'  Teacher (d_input={reg_config["d_input"]}): {n_reg:,} params')

        # Scalar divisors as buffer so they move with .to()
        self.register_buffer(
            'scalar_divisors',
            torch.tensor(_SCALAR_DIVISORS, dtype=torch.float32))

    def forward(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        """Compute fully differentiable reward from soft one-hot sequences.

        Args:
            soft_onehot: [B, 197, 4] from Gumbel-softmax.

        Returns:
            reward: [B] log2FC predictions (higher = more stable).
        """
        B, L, _ = soft_onehot.shape

        # Step 1: Straight-through for sequence channels
        seq_st = _straight_through(soft_onehot)  # [B, L, 4]

        # Step 2: KernaFold forward (differentiable)
        if self.kernafold_variant == 'bpp':
            # BPP variant needs valid_pair_mask (computed from hard sequence)
            with torch.no_grad():
                valid_pair_mask = self.kernafold._compute_valid_pair_mask(seq_st)
            kf_out = self.kernafold(seq_st, valid_pair_mask)
        else:
            kf_out = self.kernafold(seq_st)

        # Step 3: Straight-through for structure one-hots
        mfe_st = _straight_through_logits(kf_out['mfe_logits'])       # [B, L, 3]
        cent_st = _straight_through_logits(kf_out['centroid_logits'])  # [B, L, 3]

        # Step 4: Scalar normalization conversion
        # KernaFold z-normalized → raw units → teacher normalization
        scalars_raw = self.kernafold.denormalize_scalars(
            kf_out['scalars_normed'])                          # [B, 6]
        teacher_scalars = scalars_raw / self.scalar_divisors.to(scalars_raw.device)  # [B, 6]
        teacher_scalars = teacher_scalars.unsqueeze(1).expand(
            -1, L, -1)                                         # [B, L, 6]

        # Step 5: Assemble 16-channel input
        regressor_input = torch.cat([
            seq_st,           # [B, L, 4]  channels 0-3
            mfe_st,           # [B, L, 3]  channels 4-6
            cent_st,          # [B, L, 3]  channels 7-9
            teacher_scalars,  # [B, L, 6]  channels 10-15
        ], dim=-1)  # [B, L, 16]

        # Step 6: Teacher forward (frozen weights, input gradients flow)
        reward = self.regressor(regressor_input)  # [B]
        return reward


def _straight_through(soft: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: hard forward, soft backward."""
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).float()
    return soft + (hard - soft).detach()


def _straight_through_logits(logits: torch.Tensor) -> torch.Tensor:
    """Straight-through on logits: softmax for gradient, argmax for value."""
    soft = F.softmax(logits, dim=-1)
    hard = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()
    return soft + (hard - soft).detach()


if __name__ == '__main__':
    # Quick gradient flow test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapper = KernaFoldRewardWrapper(
        kernafold_checkpoint_path=(
            '/mnt/ssd1/code/kernafold/runs/bpp_v5_continuous_full/'
            'checkpoints/best.pt'),
        regressor_checkpoint_dir=(
            '/mnt/ssd1/code/narry_kim_2025/models/checkpoints/'
            'mamba_rnet_ablation_single_linear_head_lr1e-04_d256_L8_'
            'kfold5_genome'),
        device=device,
    )

    x = torch.randn(2, 197, 4, device=device, requires_grad=True)
    x_soft = F.softmax(x, dim=-1)
    reward = wrapper(x_soft)
    print(f'Reward shape: {reward.shape}, values: {reward.detach().cpu().numpy()}')

    loss = -reward.mean()
    loss.backward()
    print(f'Input grad norm: {x.grad.norm().item():.6f}')
    print(f'Gradient flows: {x.grad.abs().sum().item() > 0}')
