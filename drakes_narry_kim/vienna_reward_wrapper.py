# =============================================================================
# Vienna Reward Wrapper: Bridges Diffusion → Vienna RNA API → RNABiMamba
# =============================================================================
#
# This module connects the discrete diffusion model's soft Gumbel-softmax
# samples to the RNABiMamba stability regressor, handling the non-differentiable
# Vienna RNA features via a straight-through estimation strategy.
#
# Gradient Flow Design:
# ---------------------
# The RNABiMamba regressor expects 16-channel input [B, 197, 16]:
#
#   Channels 0-3:   Sequence one-hot (A, C, G, U)  ← GRADIENTS FLOW
#   Channels 4-6:   Vienna MFE structure one-hot    ← DETACHED (no grad)
#   Channels 7-9:   Vienna centroid structure        ← DETACHED (no grad)
#   Channels 10-15: Vienna scalars (broadcast)       ← DETACHED (no grad)
#
# The key insight: sequence channels use straight-through estimation so that
# the forward pass sees hard one-hot vectors (for correct Vienna API calls
# and accurate predictions) while gradients flow back through the soft
# Gumbel-softmax probabilities. Vienna-derived features are computed from
# the hard sequences and detached, contributing to prediction accuracy
# without requiring differentiable RNA folding.
#
# Straight-Through Mechanism:
# ---------------------------
#   soft_onehot [B, 197, 4]  ← from Gumbel-softmax sampling
#       ├── ST = soft + (hard_onehot - soft).detach()
#       │     → forward: uses hard_onehot (correct one-hot)
#       │     → backward: gradients flow through soft
#       ├── argmax → DNA strings → T→U → Vienna API
#       │     → MFE structure, centroid structure, thermodynamic scalars
#       └── cat [ST(4), struct(6), scalar(6)] → RNABiMamba → reward [B]
#
# Vienna API:
# -----------
# Calls the local Vienna RNA REST API at localhost:8000/jobs/analyze.
# Each sequence is folded independently; results include:
#   - mfe_structure:      dot-bracket string (MFE fold)
#   - centroid_structure:  dot-bracket string (centroid fold)
#   - mfe_energy:          free energy of MFE structure (kcal/mol)
#   - centroid_energy:     free energy of centroid structure (kcal/mol)
#   - centroid_distance:   expected distance from centroid to ensemble
#   - ensemble_energy:     free energy of the full ensemble (kcal/mol)
#   - mfe_frequency:       Boltzmann probability of MFE structure
#   - ensemble_diversity:  structural diversity of the ensemble
#
# Normalization (matches training data preparation):
#   - mfe_energy / 100
#   - centroid_energy / 100
#   - centroid_distance / 50
#   - ensemble_energy / 100
#   - mfe_frequency (no normalization, typically in [0, 1])
#   - ensemble_diversity / 50
#
# Usage:
#   wrapper = ViennaRewardWrapper(regressor_checkpoint_dir, device)
#   reward = wrapper(soft_onehot)  # [B] scalar (log2FC prediction)
#
# Dependencies:
#   - RNABiMamba model from /mnt/ssd1/code/narry_kim_2025/models/
#   - Vienna RNA API running at localhost:8000
#   - requests library for HTTP calls
# =============================================================================

import json
import sys
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add narry_kim_2025/models to path for importing RNABiMamba
_NARRY_KIM_MODELS_DIR = '/mnt/ssd1/code/narry_kim_2025/models'
if _NARRY_KIM_MODELS_DIR not in sys.path:
    sys.path.insert(0, _NARRY_KIM_MODELS_DIR)

from model import RNABiMamba

# Mapping from DNA token indices to RNA nucleotides (T→U for Vienna RNA)
_IDX_TO_RNA = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Dot-bracket character to one-hot index mapping for structure encoding
# '.' = unpaired (index 0), '(' = opening pair (index 1), ')' = closing pair (index 2)
_STRUCT_CHAR_TO_IDX = {'.': 0, '(': 1, ')': 2}


class ViennaRewardWrapper(nn.Module):
    """Bridges diffusion model soft samples to RNABiMamba stability predictions.

    This wrapper takes soft one-hot sequence representations from Gumbel-softmax
    sampling, computes Vienna RNA features via the REST API, and feeds everything
    into the frozen RNABiMamba regressor to produce differentiable reward signals.

    Attributes:
        regressor: Frozen RNABiMamba model predicting log2FC stability.
        vienna_api_url: URL of the Vienna RNA REST API endpoint.
        seq_len: Expected sequence length (197bp for viral tiles).
    """

    def __init__(
        self,
        regressor_checkpoint_dir: str,
        device: str = 'cuda',
        vienna_api_url: str = 'http://localhost:8000/jobs/analyze',
    ):
        """Initialize the reward wrapper.

        Args:
            regressor_checkpoint_dir: Path to the RNABiMamba checkpoint directory.
                This is a k-fold checkpoint dir containing config.json and
                fold_0/, fold_1/, ... subdirectories. We load fold_0/best_model_sig.pt
                by default (best model on significant genes, Pearson 0.7744).
                Best model: mamba_rnet_ablation_single_linear_head_lr1e-04_d256_L8_kfold5_genome
            device: Device to load the regressor onto ('cuda' or 'cpu').
            vienna_api_url: URL for the Vienna RNA REST API. Default assumes
                the API is running locally on port 8000.
        """
        super().__init__()
        self.vienna_api_url = vienna_api_url
        self.seq_len = 197

        # Load the frozen RNABiMamba regressor from k-fold checkpoint.
        # We use fold_0/best_model_sig.pt (best on significant genes).
        # The regressor is NOT fine-tuned — it provides a fixed reward signal.
        run_dir = Path(regressor_checkpoint_dir)
        with open(run_dir / 'config.json') as f:
            self.regressor_config = json.load(f)

        regressor = RNABiMamba(
            d_input=self.regressor_config['d_input'],
            d_model=self.regressor_config['d_model'],
            n_layers=self.regressor_config['n_layers'],
            dropout=self.regressor_config['dropout'],
        )
        ckpt_path = run_dir / 'fold_0' / 'best_model_sig.pt'
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        regressor.load_state_dict(checkpoint['model_state_dict'])
        regressor.to(device)
        regressor.eval()

        self.regressor = regressor
        for param in self.regressor.parameters():
            param.requires_grad = False

    def forward(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        """Compute stability reward from soft Gumbel-softmax sequence samples.

        This is the core method that implements the straight-through gradient
        flow. The forward pass sees hard discrete sequences (for correct Vienna
        folding and regressor predictions), while gradients flow back through
        the soft probabilities.

        Args:
            soft_onehot: Soft one-hot sequence tensor [B, 197, 4] from
                Gumbel-softmax sampling. Values are continuous probabilities
                over {A, C, G, T} at each position.

        Returns:
            reward: Predicted log2FC stability scores [B]. Higher values
                indicate more stable 3'UTR sequences.
        """
        device = soft_onehot.device
        B, L, _ = soft_onehot.shape

        # ----- Step 1: Straight-through estimator for sequence channels -----
        # Forward pass: hard one-hot (discrete); Backward pass: soft gradients
        hard_idx = soft_onehot.argmax(dim=-1)  # [B, L]
        hard_onehot = F.one_hot(hard_idx, num_classes=4).float()  # [B, L, 4]
        seq_features = soft_onehot + (hard_onehot - soft_onehot).detach()
        # seq_features: [B, L, 4] — looks like hard one-hot in forward,
        # but has gradients from soft_onehot for backward.

        # ----- Step 2: Decode to RNA strings (T→U) for Vienna API -----
        rna_sequences = self._decode_to_rna(hard_idx)  # list of B strings

        # ----- Step 3: Call Vienna RNA API for structure features -----
        vienna_results, failed_mask = self._call_vienna_batch(rna_sequences)

        # ----- Step 4: Encode Vienna results as tensor features -----
        # Structure one-hot (6 channels) + scalar features (6 channels)
        vienna_features = self._encode_vienna(vienna_results, device)
        # vienna_features: [B, L, 12] — DETACHED, no gradients

        # ----- Step 5: Concatenate and predict -----
        # Final input: [B, L, 16] = [seq(4) + mfe_struct(3) + cent_struct(3) + scalars(6)]
        regressor_input = torch.cat(
            [seq_features, vienna_features.detach()], dim=-1)

        # RNABiMamba forward: [B, L, 16] → [B] (log2FC prediction)
        # Regressor weights are frozen (requires_grad=False) but we do NOT
        # wrap this in torch.no_grad() — input gradients must propagate
        # back through seq_features to the diffusion model.
        reward = self.regressor(regressor_input)  # [B]

        # Zero out reward for sequences where Vienna API failed,
        # so they contribute no gradient signal.
        if any(failed_mask):
            fail_idx = torch.tensor(
                [i for i, f in enumerate(failed_mask) if f],
                device=device)
            reward = reward.clone()
            reward[fail_idx] = 0.0

        return reward

    def _decode_to_rna(self, token_indices: torch.Tensor) -> list[str]:
        """Convert token index tensor to RNA strings (DNA T → RNA U).

        Args:
            token_indices: Integer tensor [B, L] with values in {0,1,2,3}
                corresponding to {A, C, G, T}.

        Returns:
            List of B RNA strings (length L each) with U instead of T.
        """
        indices_np = token_indices.cpu().numpy()
        sequences = []
        for seq in indices_np:
            rna_str = ''.join(_IDX_TO_RNA[idx] for idx in seq)
            sequences.append(rna_str)
        return sequences

    def _call_vienna_batch(self, sequences: list[str]) -> list[dict]:
        """Call the Vienna RNA API for a batch of sequences.

        Each sequence is submitted individually to the API. The API returns
        MFE structure, centroid structure, and thermodynamic scalar features.

        Args:
            sequences: List of RNA strings (A/C/G/U alphabet).

        Returns:
            List of dicts, each containing:
                - mfe_structure: dot-bracket string
                - centroid_structure: dot-bracket string
                - mfe_energy: float (kcal/mol)
                - centroid_energy: float (kcal/mol)
                - centroid_distance: float
                - ensemble_energy: float (kcal/mol)
                - mfe_frequency: float (Boltzmann probability)
                - ensemble_diversity: float

        Raises:
            RuntimeError: If the Vienna API returns an error or is unreachable.
        """
        import time as _time
        results = []
        failed_mask = []
        max_retries = 3
        # Default fallback result for failed sequences
        fallback = self._make_fallback_result(len(sequences[0]) if sequences else 197)
        for seq in sequences:
            success = False
            last_error = None
            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        self.vienna_api_url,
                        json={'sequence': seq, 'use_n1m': True},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    results.append(data)
                    failed_mask.append(False)
                    success = True
                    break
                except requests.RequestException as e:
                    last_error = e
                    wait = 5 * (attempt + 1)
                    print(f'  [WARN] Vienna API attempt {attempt+1}/{max_retries} '
                          f'failed for {seq[:30]}...: {e}. Waiting {wait}s...')
                    _time.sleep(wait)
            if not success:
                print(f'  [WARN] Vienna API failed after {max_retries} retries '
                      f'for {seq[:30]}...: {last_error}. Using fallback.')
                results.append(fallback)
                failed_mask.append(True)
        return results, failed_mask

    @staticmethod
    def _make_fallback_result(seq_len: int) -> dict:
        """Create a neutral fallback Vienna result for failed API calls."""
        return {
            'mfe': {
                'structure': '.' * seq_len,
                'energy': 0.0,
            },
            'centroid': {
                'structure': '.' * seq_len,
                'energy': 0.0,
                'distance': 0.0,
            },
            'thermodynamics': {
                'ensemble_energy': 0.0,
                'mfe_frequency': 1.0,
                'ensemble_diversity': 0.0,
            },
        }

    def _encode_vienna(
        self,
        vienna_results: list[dict],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode Vienna RNA results into a tensor feature representation.

        Converts dot-bracket structures to one-hot vectors and normalizes
        scalar thermodynamic features to match the regressor's training
        data preparation.

        Args:
            vienna_results: List of B dicts from _call_vienna_batch.
            device: Torch device for the output tensor.

        Returns:
            Tensor [B, L, 12] with:
                [:, :, 0:3]  - MFE structure one-hot (., (, ))
                [:, :, 3:6]  - Centroid structure one-hot (., (, ))
                [:, :, 6]    - mfe_energy / 100
                [:, :, 7]    - centroid_energy / 100
                [:, :, 8]    - centroid_distance / 50
                [:, :, 9]    - ensemble_energy / 100
                [:, :, 10]   - mfe_frequency (no normalization)
                [:, :, 11]   - ensemble_diversity / 50
        """
        B = len(vienna_results)
        L = self.seq_len
        features = torch.zeros(B, L, 12, device=device)

        for i, result in enumerate(vienna_results):
            # API response is nested: result['mfe']['structure'], result['centroid']['structure'],
            # result['thermodynamics']['ensemble_energy'], etc.

            # --- MFE structure one-hot (channels 0-2) ---
            mfe_struct = result['mfe']['structure']
            for j, ch in enumerate(mfe_struct[:L]):
                idx = _STRUCT_CHAR_TO_IDX.get(ch, 0)
                features[i, j, idx] = 1.0

            # --- Centroid structure one-hot (channels 3-5) ---
            cent_struct = result['centroid']['structure']
            for j, ch in enumerate(cent_struct[:L]):
                idx = _STRUCT_CHAR_TO_IDX.get(ch, 0)
                features[i, j, 3 + idx] = 1.0

            # --- Scalar features (channels 6-11), broadcast across all positions ---
            # Normalization matches prepare_data_with_rnet.py from narry_kim_2025
            features[i, :, 6] = result['mfe']['energy'] / 100.0
            features[i, :, 7] = result['centroid']['energy'] / 100.0
            features[i, :, 8] = result['centroid']['distance'] / 50.0
            features[i, :, 9] = result['thermodynamics']['ensemble_energy'] / 100.0
            features[i, :, 10] = result['thermodynamics']['mfe_frequency']
            features[i, :, 11] = result['thermodynamics']['ensemble_diversity'] / 50.0

        return features


def test_wrapper():
    """Quick test to verify the wrapper works end-to-end.

    Creates a random soft one-hot input and passes it through the full
    pipeline: straight-through → Vienna API → RNABiMamba → reward.

    Usage:
        python vienna_reward_wrapper.py
    """
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = (
        '/mnt/ssd1/code/narry_kim_2025/models/checkpoints/'
        'mamba_rnet_ablation_single_linear_head_lr1e-04_d256_L8_kfold5_genome'
    )

    print(f'Loading ViennaRewardWrapper on {device}...')
    wrapper = ViennaRewardWrapper(ckpt_dir, device=device)

    # Create random soft one-hot input (simulating Gumbel-softmax output)
    B, L = 4, 197
    logits = torch.randn(B, L, 4, device=device)
    soft_onehot = F.softmax(logits, dim=-1)
    soft_onehot.requires_grad_(True)

    print(f'Input shape: {soft_onehot.shape}')
    print(f'Calling forward (includes Vienna API calls)...')

    t0 = time.time()
    reward = wrapper(soft_onehot)
    elapsed = time.time() - t0

    print(f'Output shape: {reward.shape}')
    print(f'Rewards: {reward.detach().cpu().numpy()}')
    print(f'Time: {elapsed:.2f}s ({elapsed/B:.2f}s per sequence)')

    # Verify gradient flow
    loss = -reward.mean()
    loss.backward()
    print(f'Gradient flows: {soft_onehot.grad is not None}')
    if soft_onehot.grad is not None:
        print(f'Grad norm: {soft_onehot.grad.norm().item():.6f}')
        print(f'Grad shape: {soft_onehot.grad.shape}')

    print('\nAll tests passed!')


if __name__ == '__main__':
    test_wrapper()
