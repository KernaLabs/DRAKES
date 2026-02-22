#!/usr/bin/env python3
"""
Diagnostic script: Probe why the RNABiMamba regressor gives higher scores
to sequences with worse structural properties after DRAKES fine-tuning.

Experiments:
  A) Channel swap: fine-tuned sequences + baseline structure channels
  B) Reverse swap: baseline sequences + fine-tuned structure channels
  C) Ablation: zero out sequence vs structure channels independently

Usage:
    python probe_regressor.py
"""

import datetime
import json
import os
import random
import string
import sys
import time

import numpy as np
import omegaconf
import requests
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

# ---- Hydra custom resolvers (must be registered before compose()) ----
omegaconf.OmegaConf.register_new_resolver(
    "uuid",
    lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))
    + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
    use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

# Must be after resolver registration
import diffusion as diffusion_module

import drakes_paths as dp
RNABiMamba = dp.import_rnabimamba()

# =============================================================================
# Configuration
# =============================================================================

REGRESSOR_CKPT_DIR = str(dp.narry_kim.regressor_ckpt_dir)
PRETRAINED_CKPT = str(dp.narry_kim.experiments_dir / 'checkpoints' / 'best.ckpt')
FINETUNED_CKPT = None  # Set to a specific run path when needed
VIENNA_API_URL = 'http://localhost:8000/jobs/analyze'
NUM_SEQS = 32
NUM_STEPS = 128
SEQ_LEN = 197
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Nucleotide mappings
IDX_TO_NT = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
IDX_TO_RNA = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
NT_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
STRUCT_CHAR_TO_IDX = {'.': 0, '(': 1, ')': 2}


# =============================================================================
# Helper functions
# =============================================================================

def load_regressor(ckpt_dir, device):
    """Load the frozen RNABiMamba regressor."""
    run_dir = Path(ckpt_dir)
    with open(run_dir / 'config.json') as f:
        config = json.load(f)
    regressor = RNABiMamba(
        d_input=config['d_input'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
    )
    ckpt_path = run_dir / 'fold_0' / 'best_model_sig.pt'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.to(device)
    regressor.eval()
    for param in regressor.parameters():
        param.requires_grad = False
    print(f'  Loaded regressor from {ckpt_path}')
    print(f'  Config: d_input={config["d_input"]}, d_model={config["d_model"]}, '
          f'n_layers={config["n_layers"]}')
    return regressor


def load_diffusion_model(cfg, checkpoint_path, device):
    """Load a diffusion model from a Lightning or state_dict checkpoint."""
    # First try loading as Lightning checkpoint
    try:
        model = diffusion_module.Diffusion.load_from_checkpoint(
            checkpoint_path, config=cfg)
        model.to(device)
        print(f'  Loaded as Lightning checkpoint: {checkpoint_path}')
        return model
    except Exception as e:
        print(f'  Lightning load failed ({e}), trying state_dict...')

    # Fall back to state_dict loading: instantiate a fresh model, then load weights
    model = diffusion_module.Diffusion(config=cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f'  Loaded as state_dict: {checkpoint_path}')
    return model


def generate_sequences(model, num_seqs, num_steps, device):
    """Generate sequences using DDPM sampling (no gradient)."""
    model.eval()
    all_indices = []  # list of [B, L] tensors
    remaining = num_seqs
    while remaining > 0:
        bsz = min(remaining, 32)
        with torch.no_grad():
            x = model.mask_index * torch.ones(
                bsz, SEQ_LEN, dtype=torch.int64, device=device)
            timesteps = torch.linspace(1, 1e-3, num_steps + 1, device=device)
            dt = (1 - 1e-3) / num_steps
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                x = model._ddpm_update(x, t, dt)
        all_indices.append(x.cpu())
        remaining -= bsz
    indices = torch.cat(all_indices, dim=0)[:num_seqs]  # [N, L]

    # Convert to DNA strings (replace mask token with A as fallback)
    sequences = []
    for i in range(indices.shape[0]):
        seq = ''.join(IDX_TO_NT.get(indices[i, j].item(), 'A') for j in range(SEQ_LEN))
        sequences.append(seq)
    return sequences, indices


def seqs_to_onehot(sequences, device):
    """Convert list of DNA strings to one-hot tensor [B, L, 4]."""
    B = len(sequences)
    oh = torch.zeros(B, SEQ_LEN, 4, device=device)
    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq):
            oh[i, j, NT_TO_IDX.get(ch, 0)] = 1.0
    return oh


def call_vienna_single(sequence, max_retries=3):
    """Call Vienna RNA API for a single RNA sequence."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                VIENNA_API_URL,
                json={'sequence': sequence, 'use_n1m': True},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json(), False
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f'    [WARN] Vienna API attempt {attempt+1}/{max_retries} failed: {e}. '
                  f'Waiting {wait}s...')
            time.sleep(wait)
    # Return fallback
    return {
        'mfe': {'structure': '.' * len(sequence), 'energy': 0.0},
        'centroid': {'structure': '.' * len(sequence), 'energy': 0.0, 'distance': 0.0},
        'thermodynamics': {'ensemble_energy': 0.0, 'mfe_frequency': 1.0,
                           'ensemble_diversity': 0.0},
    }, True


def call_vienna_batch(dna_sequences):
    """Call Vienna RNA API for a batch of DNA sequences (T -> U conversion)."""
    results = []
    failed = []
    for i, dna_seq in enumerate(dna_sequences):
        rna_seq = dna_seq.replace('T', 'U')
        result, is_failed = call_vienna_single(rna_seq)
        results.append(result)
        failed.append(is_failed)
        if (i + 1) % 8 == 0:
            print(f'    Vienna API: {i+1}/{len(dna_sequences)} done')
    return results, failed


def encode_vienna_features(vienna_results, device):
    """Encode Vienna RNA results into tensor features [B, L, 12].

    Channels:
        0-2: MFE structure one-hot (., (, ))
        3-5: Centroid structure one-hot (., (, ))
        6:   mfe_energy / 100
        7:   centroid_energy / 100
        8:   centroid_distance / 50
        9:   ensemble_energy / 100
        10:  mfe_frequency (no normalization)
        11:  ensemble_diversity / 50
    """
    B = len(vienna_results)
    features = torch.zeros(B, SEQ_LEN, 12, device=device)

    for i, result in enumerate(vienna_results):
        # MFE structure one-hot (channels 0-2)
        mfe_struct = result['mfe']['structure']
        for j, ch in enumerate(mfe_struct[:SEQ_LEN]):
            idx = STRUCT_CHAR_TO_IDX.get(ch, 0)
            features[i, j, idx] = 1.0

        # Centroid structure one-hot (channels 3-5)
        cent_struct = result['centroid']['structure']
        for j, ch in enumerate(cent_struct[:SEQ_LEN]):
            idx = STRUCT_CHAR_TO_IDX.get(ch, 0)
            features[i, j, 3 + idx] = 1.0

        # Scalar features (channels 6-11), broadcast across all positions
        features[i, :, 6] = result['mfe']['energy'] / 100.0
        features[i, :, 7] = result['centroid']['energy'] / 100.0
        features[i, :, 8] = result['centroid']['distance'] / 50.0
        features[i, :, 9] = result['thermodynamics']['ensemble_energy'] / 100.0
        features[i, :, 10] = result['thermodynamics']['mfe_frequency']
        features[i, :, 11] = result['thermodynamics']['ensemble_diversity'] / 50.0

    return features


def build_full_input(seq_onehot, vienna_features):
    """Concatenate sequence one-hot [B,L,4] with vienna features [B,L,12] -> [B,L,16]."""
    return torch.cat([seq_onehot, vienna_features], dim=-1)


def predict_rewards(regressor, full_input):
    """Run regressor and return predictions [B]."""
    with torch.no_grad():
        return regressor(full_input)


def compute_composition(sequences):
    """Compute nucleotide composition statistics."""
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    total = 0
    for seq in sequences:
        for ch in seq:
            if ch in counts:
                counts[ch] += 1
                total += 1
    fracs = {k: v / total for k, v in counts.items()}
    fracs['GC'] = fracs['G'] + fracs['C']
    return fracs


def summarize_vienna_scalars(vienna_results):
    """Compute mean scalar features from Vienna results."""
    mfe_energies = [r['mfe']['energy'] for r in vienna_results]
    cent_energies = [r['centroid']['energy'] for r in vienna_results]
    ens_energies = [r['thermodynamics']['ensemble_energy'] for r in vienna_results]
    mfe_freqs = [r['thermodynamics']['mfe_frequency'] for r in vienna_results]
    cent_dists = [r['centroid']['distance'] for r in vienna_results]
    ens_divs = [r['thermodynamics']['ensemble_diversity'] for r in vienna_results]

    # Count paired bases in MFE structure
    paired_fracs = []
    for r in vienna_results:
        struct = r['mfe']['structure']
        n_paired = struct.count('(') + struct.count(')')
        paired_fracs.append(n_paired / len(struct))

    return {
        'mfe_energy': np.mean(mfe_energies),
        'centroid_energy': np.mean(cent_energies),
        'ensemble_energy': np.mean(ens_energies),
        'mfe_frequency': np.mean(mfe_freqs),
        'centroid_distance': np.mean(cent_dists),
        'ensemble_diversity': np.mean(ens_divs),
        'paired_frac': np.mean(paired_fracs),
    }


def print_separator(title):
    """Print a formatted section separator."""
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')


# =============================================================================
# Main diagnostic script
# =============================================================================

def main():
    print_separator('REGRESSOR DIAGNOSTIC: Probing Sequence vs Structure Sensitivity')
    print(f'  Device: {DEVICE}')
    print(f'  Num sequences: {NUM_SEQS}')
    print(f'  Num diffusion steps: {NUM_STEPS}')

    # ---- Step 0: Initialize Hydra and load config ----
    print_separator('Step 0: Loading configuration')
    GlobalHydra.instance().clear()
    initialize(config_path="configs", job_name="probe_regressor")
    cfg = compose(config_name="config.yaml")

    # ---- Step 1: Load regressor ----
    print_separator('Step 1: Loading RNABiMamba regressor')
    regressor = load_regressor(REGRESSOR_CKPT_DIR, DEVICE)

    # ---- Step 2: Load pretrained diffusion model and generate baseline sequences ----
    print_separator('Step 2: Loading pretrained model and generating baseline sequences')
    cfg_baseline = cfg.copy()
    cfg_baseline.eval.checkpoint_path = PRETRAINED_CKPT
    pretrained_model = load_diffusion_model(cfg_baseline, PRETRAINED_CKPT, DEVICE)

    print(f'  Generating {NUM_SEQS} baseline sequences...')
    baseline_seqs, baseline_idx = generate_sequences(
        pretrained_model, NUM_SEQS, NUM_STEPS, DEVICE)
    del pretrained_model
    torch.cuda.empty_cache()

    baseline_comp = compute_composition(baseline_seqs)
    print(f'  Baseline composition: A={baseline_comp["A"]:.3f} C={baseline_comp["C"]:.3f} '
          f'G={baseline_comp["G"]:.3f} T={baseline_comp["T"]:.3f} GC={baseline_comp["GC"]:.3f}')

    # ---- Step 3: Load fine-tuned model and generate fine-tuned sequences ----
    print_separator('Step 3: Loading fine-tuned model and generating sequences')

    # Load pretrained first, then overwrite with fine-tuned state_dict
    cfg_ft = cfg.copy()
    cfg_ft.eval.checkpoint_path = PRETRAINED_CKPT
    finetuned_model = load_diffusion_model(cfg_ft, PRETRAINED_CKPT, DEVICE)
    ft_state_dict = torch.load(FINETUNED_CKPT, map_location=DEVICE, weights_only=True)
    finetuned_model.load_state_dict(ft_state_dict)
    finetuned_model.to(DEVICE)
    print(f'  Loaded fine-tuned weights from {FINETUNED_CKPT}')

    print(f'  Generating {NUM_SEQS} fine-tuned sequences...')
    finetuned_seqs, finetuned_idx = generate_sequences(
        finetuned_model, NUM_SEQS, NUM_STEPS, DEVICE)
    del finetuned_model
    torch.cuda.empty_cache()

    finetuned_comp = compute_composition(finetuned_seqs)
    print(f'  Fine-tuned composition: A={finetuned_comp["A"]:.3f} C={finetuned_comp["C"]:.3f} '
          f'G={finetuned_comp["G"]:.3f} T={finetuned_comp["T"]:.3f} GC={finetuned_comp["GC"]:.3f}')

    # ---- Step 4: Call Vienna RNA API for both sets ----
    print_separator('Step 4: Calling Vienna RNA API')

    print('  Folding baseline sequences...')
    baseline_vienna, baseline_failed = call_vienna_batch(baseline_seqs)
    n_base_fail = sum(baseline_failed)
    if n_base_fail:
        print(f'  WARNING: {n_base_fail} baseline sequences failed Vienna API')

    print('  Folding fine-tuned sequences...')
    finetuned_vienna, finetuned_failed = call_vienna_batch(finetuned_seqs)
    n_ft_fail = sum(finetuned_failed)
    if n_ft_fail:
        print(f'  WARNING: {n_ft_fail} fine-tuned sequences failed Vienna API')

    # Summarize Vienna scalars
    base_scalars = summarize_vienna_scalars(baseline_vienna)
    ft_scalars = summarize_vienna_scalars(finetuned_vienna)

    print('\n  Vienna scalar comparison:')
    print(f'  {"Metric":<25s} {"Baseline":>12s} {"Fine-tuned":>12s} {"Delta":>12s}')
    print(f'  {"-"*61}')
    for key in base_scalars:
        delta = ft_scalars[key] - base_scalars[key]
        print(f'  {key:<25s} {base_scalars[key]:>12.4f} {ft_scalars[key]:>12.4f} {delta:>+12.4f}')

    # ---- Step 5: Build full 16-channel inputs ----
    print_separator('Step 5: Building 16-channel inputs and getting baseline predictions')

    baseline_oh = seqs_to_onehot(baseline_seqs, DEVICE)    # [N, L, 4]
    finetuned_oh = seqs_to_onehot(finetuned_seqs, DEVICE)  # [N, L, 4]

    baseline_vfeats = encode_vienna_features(baseline_vienna, DEVICE)    # [N, L, 12]
    finetuned_vfeats = encode_vienna_features(finetuned_vienna, DEVICE)  # [N, L, 12]

    # Full inputs: [N, L, 16]
    baseline_full = build_full_input(baseline_oh, baseline_vfeats)
    finetuned_full = build_full_input(finetuned_oh, finetuned_vfeats)

    # Get predictions for both sets with their own features
    baseline_rewards = predict_rewards(regressor, baseline_full).cpu().numpy()
    finetuned_rewards = predict_rewards(regressor, finetuned_full).cpu().numpy()

    print(f'\n  Baseline sequences:  mean reward = {np.mean(baseline_rewards):.4f} '
          f'+/- {np.std(baseline_rewards):.4f}  (min={np.min(baseline_rewards):.4f}, '
          f'max={np.max(baseline_rewards):.4f})')
    print(f'  Fine-tuned sequences: mean reward = {np.mean(finetuned_rewards):.4f} '
          f'+/- {np.std(finetuned_rewards):.4f}  (min={np.min(finetuned_rewards):.4f}, '
          f'max={np.max(finetuned_rewards):.4f})')
    print(f'  Delta (FT - Base): {np.mean(finetuned_rewards) - np.mean(baseline_rewards):+.4f}')

    # ==========================================================================
    # EXPERIMENT A: Channel swap - fine-tuned seq + baseline structure
    # ==========================================================================
    print_separator('Experiment A: Channel Swap (fine-tuned seq + baseline structure)')
    print('  Question: If we give the regressor fine-tuned sequences but baseline')
    print('  structural features, does the reward stay high?')
    print('  If YES -> regressor is driven by sequence identity, not structure.')

    swap_a_input = build_full_input(finetuned_oh, baseline_vfeats)
    swap_a_rewards = predict_rewards(regressor, swap_a_input).cpu().numpy()

    print(f'\n  Fine-tuned seq + baseline struct: mean reward = {np.mean(swap_a_rewards):.4f} '
          f'+/- {np.std(swap_a_rewards):.4f}')
    print(f'  vs. fine-tuned seq + own struct:  mean reward = {np.mean(finetuned_rewards):.4f}')
    print(f'  vs. baseline seq + own struct:    mean reward = {np.mean(baseline_rewards):.4f}')
    delta_a = np.mean(swap_a_rewards) - np.mean(finetuned_rewards)
    print(f'  Change from swapping structure:   {delta_a:+.4f}')
    if abs(delta_a) < 0.1 * abs(np.mean(finetuned_rewards) - np.mean(baseline_rewards)):
        print('  >> FINDING: Structure swap has SMALL effect. Reward is mainly driven by SEQUENCE.')
    else:
        print('  >> FINDING: Structure swap has SIGNIFICANT effect. Structure matters.')

    # ==========================================================================
    # EXPERIMENT B: Reverse swap - baseline seq + fine-tuned structure
    # ==========================================================================
    print_separator('Experiment B: Reverse Swap (baseline seq + fine-tuned structure)')
    print('  Question: If we give the regressor baseline sequences but fine-tuned')
    print('  structural features, does the reward go up?')
    print('  If YES -> structural features alone drive the reward increase.')

    swap_b_input = build_full_input(baseline_oh, finetuned_vfeats)
    swap_b_rewards = predict_rewards(regressor, swap_b_input).cpu().numpy()

    print(f'\n  Baseline seq + fine-tuned struct: mean reward = {np.mean(swap_b_rewards):.4f} '
          f'+/- {np.std(swap_b_rewards):.4f}')
    print(f'  vs. baseline seq + own struct:    mean reward = {np.mean(baseline_rewards):.4f}')
    print(f'  vs. fine-tuned seq + own struct:  mean reward = {np.mean(finetuned_rewards):.4f}')
    delta_b = np.mean(swap_b_rewards) - np.mean(baseline_rewards)
    print(f'  Change from swapping structure:   {delta_b:+.4f}')
    if abs(delta_b) < 0.1 * abs(np.mean(finetuned_rewards) - np.mean(baseline_rewards)):
        print('  >> FINDING: Swapping in FT structure has SMALL effect. Structure is NOT the driver.')
    else:
        print('  >> FINDING: Swapping in FT structure has SIGNIFICANT effect.')

    # ==========================================================================
    # EXPERIMENT C: Ablation - zero out channels
    # ==========================================================================
    print_separator('Experiment C: Channel Ablation (zero out seq vs structure)')
    print('  For fine-tuned sequences:')
    print('    C1: Zero out sequence channels (0:4), keep structure (4:16)')
    print('    C2: Zero out structure channels (4:16), keep sequence (0:4)')
    print('    C3: Zero out only scalar channels (10:16), keep seq + struct one-hot')
    print('    C4: Zero out only struct one-hot (4:10), keep seq + scalars')

    # C1: Zero sequence, keep structure
    c1_input = finetuned_full.clone()
    c1_input[:, :, :4] = 0.0
    c1_rewards = predict_rewards(regressor, c1_input).cpu().numpy()

    # C2: Zero structure, keep sequence
    c2_input = finetuned_full.clone()
    c2_input[:, :, 4:] = 0.0
    c2_rewards = predict_rewards(regressor, c2_input).cpu().numpy()

    # C3: Zero scalars only (channels 10-15), keep seq + struct one-hot
    c3_input = finetuned_full.clone()
    c3_input[:, :, 10:] = 0.0
    c3_rewards = predict_rewards(regressor, c3_input).cpu().numpy()

    # C4: Zero struct one-hot only (channels 4-9), keep seq + scalars
    c4_input = finetuned_full.clone()
    c4_input[:, :, 4:10] = 0.0
    c4_rewards = predict_rewards(regressor, c4_input).cpu().numpy()

    print(f'\n  {"Condition":<45s} {"Mean reward":>12s} {"Std":>8s}')
    print(f'  {"-"*65}')
    print(f'  {"Full input (reference)":45s} {np.mean(finetuned_rewards):>12.4f} {np.std(finetuned_rewards):>8.4f}')
    print(f'  {"C1: zero seq (0:4), keep struct (4:16)":45s} {np.mean(c1_rewards):>12.4f} {np.std(c1_rewards):>8.4f}')
    print(f'  {"C2: zero struct (4:16), keep seq (0:4)":45s} {np.mean(c2_rewards):>12.4f} {np.std(c2_rewards):>8.4f}')
    print(f'  {"C3: zero scalars (10:16), keep seq+struct_oh":45s} {np.mean(c3_rewards):>12.4f} {np.std(c3_rewards):>8.4f}')
    print(f'  {"C4: zero struct_oh (4:10), keep seq+scalars":45s} {np.mean(c4_rewards):>12.4f} {np.std(c4_rewards):>8.4f}')

    # Also run ablation on baseline sequences for comparison
    print('\n  For baseline sequences:')

    c1b_input = baseline_full.clone()
    c1b_input[:, :, :4] = 0.0
    c1b_rewards = predict_rewards(regressor, c1b_input).cpu().numpy()

    c2b_input = baseline_full.clone()
    c2b_input[:, :, 4:] = 0.0
    c2b_rewards = predict_rewards(regressor, c2b_input).cpu().numpy()

    c3b_input = baseline_full.clone()
    c3b_input[:, :, 10:] = 0.0
    c3b_rewards = predict_rewards(regressor, c3b_input).cpu().numpy()

    c4b_input = baseline_full.clone()
    c4b_input[:, :, 4:10] = 0.0
    c4b_rewards = predict_rewards(regressor, c4b_input).cpu().numpy()

    print(f'  {"Condition":<45s} {"Mean reward":>12s} {"Std":>8s}')
    print(f'  {"-"*65}')
    print(f'  {"Full input (reference)":45s} {np.mean(baseline_rewards):>12.4f} {np.std(baseline_rewards):>8.4f}')
    print(f'  {"C1: zero seq (0:4), keep struct (4:16)":45s} {np.mean(c1b_rewards):>12.4f} {np.std(c1b_rewards):>8.4f}')
    print(f'  {"C2: zero struct (4:16), keep seq (0:4)":45s} {np.mean(c2b_rewards):>12.4f} {np.std(c2b_rewards):>8.4f}')
    print(f'  {"C3: zero scalars (10:16), keep seq+struct_oh":45s} {np.mean(c3b_rewards):>12.4f} {np.std(c3b_rewards):>8.4f}')
    print(f'  {"C4: zero struct_oh (4:10), keep seq+scalars":45s} {np.mean(c4b_rewards):>12.4f} {np.std(c4b_rewards):>8.4f}')

    # ==========================================================================
    # EXPERIMENT D: Per-sequence correlation analysis
    # ==========================================================================
    print_separator('Experiment D: Per-sequence correlation analysis')

    # Compute per-sequence MFE energy
    base_mfe = np.array([r['mfe']['energy'] for r in baseline_vienna])
    ft_mfe = np.array([r['mfe']['energy'] for r in finetuned_vienna])

    # Compute per-sequence A content
    base_a_frac = np.array([seq.count('A') / len(seq) for seq in baseline_seqs])
    ft_a_frac = np.array([seq.count('A') / len(seq) for seq in finetuned_seqs])

    # Compute per-sequence GC content
    base_gc = np.array([(seq.count('G') + seq.count('C')) / len(seq) for seq in baseline_seqs])
    ft_gc = np.array([(seq.count('G') + seq.count('C')) / len(seq) for seq in finetuned_seqs])

    # Correlations within fine-tuned sequences
    from scipy import stats as scipy_stats

    corr_reward_mfe, p_reward_mfe = scipy_stats.pearsonr(finetuned_rewards, ft_mfe)
    corr_reward_a, p_reward_a = scipy_stats.pearsonr(finetuned_rewards, ft_a_frac)
    corr_reward_gc, p_reward_gc = scipy_stats.pearsonr(finetuned_rewards, ft_gc)

    print('  Within fine-tuned sequences:')
    print(f'    Corr(reward, mfe_energy):  r={corr_reward_mfe:.4f}, p={p_reward_mfe:.4e}')
    print(f'    Corr(reward, A_fraction):  r={corr_reward_a:.4f}, p={p_reward_a:.4e}')
    print(f'    Corr(reward, GC_content):  r={corr_reward_gc:.4f}, p={p_reward_gc:.4e}')

    # Same for baseline
    corr_b_reward_mfe, p_b_reward_mfe = scipy_stats.pearsonr(baseline_rewards, base_mfe)
    corr_b_reward_a, p_b_reward_a = scipy_stats.pearsonr(baseline_rewards, base_a_frac)
    corr_b_reward_gc, p_b_reward_gc = scipy_stats.pearsonr(baseline_rewards, base_gc)

    print('  Within baseline sequences:')
    print(f'    Corr(reward, mfe_energy):  r={corr_b_reward_mfe:.4f}, p={p_b_reward_mfe:.4e}')
    print(f'    Corr(reward, A_fraction):  r={corr_b_reward_a:.4f}, p={p_b_reward_a:.4e}')
    print(f'    Corr(reward, GC_content):  r={corr_b_reward_gc:.4f}, p={p_b_reward_gc:.4e}')

    # ==========================================================================
    # EXPERIMENT E: Uniform baseline - what does the regressor predict for
    # random one-hot with zero structure?
    # ==========================================================================
    print_separator('Experiment E: Regressor sensitivity to A-rich sequences')
    print('  Test: Create synthetic sequences with varying A content and')
    print('  zero structure features to see if the regressor intrinsically')
    print('  prefers A-rich sequences.')

    a_fracs_to_test = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    print(f'\n  {"A fraction":>12s} {"Mean reward":>12s} {"Std":>8s}')
    print(f'  {"-"*32}')

    for target_a in a_fracs_to_test:
        # Generate synthetic sequences with controlled A content
        synth_seqs = []
        for _ in range(NUM_SEQS):
            seq_list = []
            for pos in range(SEQ_LEN):
                if random.random() < target_a:
                    seq_list.append('A')
                else:
                    seq_list.append(random.choice(['C', 'G', 'T']))
            synth_seqs.append(''.join(seq_list))

        synth_oh = seqs_to_onehot(synth_seqs, DEVICE)
        # Zero structure channels
        synth_input = torch.cat([synth_oh, torch.zeros(NUM_SEQS, SEQ_LEN, 12, device=DEVICE)], dim=-1)
        synth_rewards = predict_rewards(regressor, synth_input).cpu().numpy()
        actual_a = np.mean([s.count('A') / len(s) for s in synth_seqs])
        print(f'  {actual_a:>12.3f} {np.mean(synth_rewards):>12.4f} {np.std(synth_rewards):>8.4f}')

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_separator('SUMMARY')
    print(f'  Baseline mean reward:    {np.mean(baseline_rewards):.4f}')
    print(f'  Fine-tuned mean reward:  {np.mean(finetuned_rewards):.4f}')
    print(f'  Reward increase:         {np.mean(finetuned_rewards) - np.mean(baseline_rewards):+.4f}')
    print()
    print(f'  Exp A (FT seq + base struct):       {np.mean(swap_a_rewards):.4f}  '
          f'(change from FT: {delta_a:+.4f})')
    print(f'  Exp B (base seq + FT struct):        {np.mean(swap_b_rewards):.4f}  '
          f'(change from base: {delta_b:+.4f})')
    print()

    # Compute attribution
    reward_gap = np.mean(finetuned_rewards) - np.mean(baseline_rewards)
    seq_contribution = np.mean(swap_a_rewards) - np.mean(baseline_rewards)  # changing seq (keeping base struct)
    struct_contribution = np.mean(swap_b_rewards) - np.mean(baseline_rewards)  # changing struct (keeping base seq)
    interaction = reward_gap - seq_contribution - struct_contribution

    if abs(reward_gap) > 1e-6:
        print(f'  Attribution decomposition:')
        print(f'    Sequence channels:   {seq_contribution:+.4f}  '
              f'({100*seq_contribution/reward_gap:.1f}% of gap)')
        print(f'    Structure channels:  {struct_contribution:+.4f}  '
              f'({100*struct_contribution/reward_gap:.1f}% of gap)')
        print(f'    Interaction term:    {interaction:+.4f}  '
              f'({100*interaction/reward_gap:.1f}% of gap)')
    print()

    seq_only_gap = np.mean(c2_rewards) - np.mean(c2b_rewards)
    struct_only_gap = np.mean(c1_rewards) - np.mean(c1b_rewards)
    print(f'  Ablation (seq-only, zeroed struct):')
    print(f'    FT seq-only reward:    {np.mean(c2_rewards):.4f}')
    print(f'    Base seq-only reward:  {np.mean(c2b_rewards):.4f}')
    print(f'    Gap:                   {seq_only_gap:+.4f}')
    print()
    print(f'  Ablation (struct-only, zeroed seq):')
    print(f'    FT struct-only reward:   {np.mean(c1_rewards):.4f}')
    print(f'    Base struct-only reward: {np.mean(c1b_rewards):.4f}')
    print(f'    Gap:                     {struct_only_gap:+.4f}')
    print()

    if abs(reward_gap) > 1e-6:
        if abs(seq_contribution) > 2 * abs(struct_contribution):
            print('  CONCLUSION: Reward increase is PRIMARILY driven by SEQUENCE identity.')
            print('  The regressor has learned sequence patterns (e.g., A-richness, GC content)')
            print('  that correlate with stability in training data, and DRAKES exploits them.')
        elif abs(struct_contribution) > 2 * abs(seq_contribution):
            print('  CONCLUSION: Reward increase is PRIMARILY driven by STRUCTURE features.')
        else:
            print('  CONCLUSION: Both sequence and structure contribute to the reward increase.')
    print()


if __name__ == '__main__':
    main()
