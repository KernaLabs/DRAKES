"""Compare KernaFold predictions vs real Vienna RNA API on generated sequences.

Generates sequences from the pretrained diffusion model, runs both KernaFold
and Vienna RNA API, and compares structure accuracy, scalar correlations, and
teacher reward agreement.

Usage:
    CUDA_VISIBLE_DEVICES=1 python compare_kernafold_vienna.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

# === Imports via importlib to avoid model.py name clashes ===
import importlib.util

def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

import drakes_paths as dp
build_kernafold = dp.import_kernafold_build()
RNABiMamba = dp.import_rnabimamba()

# === Constants ===
_IDX_TO_RNA = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
_STRUCT_CHAR_TO_IDX = {'.': 0, '(': 1, ')': 2}
_SCALAR_DIVISORS = [100.0, 100.0, 50.0, 100.0, 1.0, 50.0]
_SCALAR_NAMES = ['mfe_energy', 'centroid_energy', 'centroid_distance',
                 'ensemble_energy', 'mfe_frequency', 'ensemble_diversity']

VIENNA_API_URL = 'http://localhost:8000/jobs/analyze'
KERNAFOLD_CKPT = str(dp.narry_kim.kernafold_ckpt)
TEACHER_DIR = str(dp.narry_kim.regressor_ckpt_dir)
DIFFUSION_CKPT = str(dp.narry_kim.experiments_dir / 'checkpoints' / 'best.ckpt')

N_SEQUENCES = 64
DEVICE = 'cuda'


def load_kernafold(device):
    """Load KernaFold model."""
    ckpt = torch.load(KERNAFOLD_CKPT, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = build_kernafold(
        variant=cfg['variant'], d_model=cfg['d_model'],
        n_conv_layers=cfg['n_conv_layers'],
        n_transformer_layers=cfg['n_transformer_layers'],
        n_heads=cfg['n_heads'], dropout=cfg['dropout'],
        scalar_mean=cfg.get('scalar_mean'), scalar_std=cfg.get('scalar_std'),
        d_pair=cfg.get('d_pair', 64), n_pair_layers=cfg.get('n_pair_layers', 8))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg['variant']


def load_teacher(device):
    """Load RNABiMamba teacher."""
    run_dir = Path(TEACHER_DIR)
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)
    model = RNABiMamba(d_input=cfg['d_input'], d_model=cfg['d_model'],
                       n_layers=cfg['n_layers'], dropout=cfg['dropout'])
    ckpt = torch.load(run_dir / 'fold_0' / 'best_model_sig.pt',
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def generate_sequences(n, device):
    """Generate sequences from pretrained diffusion model."""
    import os, omegaconf
    from diffusion import Diffusion
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra

    # Register custom resolvers needed by config
    for name, fn in [
        ('uuid', lambda: 'compare'),
        ('cwd', os.getcwd),
        ('device_count', torch.cuda.device_count),
        ('eval', eval),
        ('div_up', lambda x, y: (x + y - 1) // y),
    ]:
        if not omegaconf.OmegaConf.has_resolver(name):
            omegaconf.OmegaConf.register_new_resolver(name, fn, use_cache=False)

    GlobalHydra.instance().clear()
    with initialize(config_path='configs', version_base=None):
        cfg = compose(config_name='config')
    cfg.eval.checkpoint_path = DIFFUSION_CKPT

    model = Diffusion.load_from_checkpoint(DIFFUSION_CKPT, config=cfg)
    model.to(device).eval()

    all_seqs = []
    bs = 16
    for i in range(0, n, bs):
        cur_bs = min(bs, n - i)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float32):
                sample = model._sample_finetune_gradient(cur_bs)[0]
        all_seqs.append(sample)
    return torch.cat(all_seqs, dim=0)[:n]  # [N, 197, 4]


def sequences_to_rna(seq_onehot):
    """Convert one-hot [N, L, 4] to list of RNA strings."""
    indices = seq_onehot.argmax(dim=-1).cpu().numpy()
    return [''.join(_IDX_TO_RNA[idx] for idx in row) for row in indices]


def call_vienna_batch(rna_sequences):
    """Call Vienna RNA API for each sequence."""
    results = []
    for seq in rna_sequences:
        for attempt in range(3):
            try:
                resp = requests.post(VIENNA_API_URL,
                                     json={'sequence': seq, 'use_n1m': True},
                                     timeout=30)
                resp.raise_for_status()
                results.append(resp.json())
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  Vienna API failed for {seq[:30]}...: {e}')
                    results.append(None)
                time.sleep(2)
    return results


def encode_vienna_features(vienna_results, seq_len=197):
    """Encode Vienna results as 16-channel features [N, L, 16] from seq + vienna.

    Returns: (features_12ch [N, L, 12], scalars_raw [N, 6])
    """
    N = len(vienna_results)
    L = seq_len
    struct_scalar = np.zeros((N, L, 12), dtype=np.float32)
    scalars_raw = np.zeros((N, 6), dtype=np.float32)

    for i, result in enumerate(vienna_results):
        if result is None:
            continue

        # MFE structure one-hot
        mfe_struct = result['mfe']['structure']
        for j, ch in enumerate(mfe_struct[:L]):
            idx = _STRUCT_CHAR_TO_IDX.get(ch, 0)
            struct_scalar[i, j, idx] = 1.0

        # Centroid structure one-hot
        cent_struct = result['centroid']['structure']
        for j, ch in enumerate(cent_struct[:L]):
            idx = _STRUCT_CHAR_TO_IDX.get(ch, 0)
            struct_scalar[i, j, 3 + idx] = 1.0

        # Raw scalars
        raw = [
            result['mfe']['energy'],
            result['centroid']['energy'],
            result['centroid']['distance'],
            result['thermodynamics']['ensemble_energy'],
            result['thermodynamics']['mfe_frequency'],
            result['thermodynamics']['ensemble_diversity'],
        ]
        scalars_raw[i] = raw

        # Normalized scalars broadcast
        struct_scalar[i, :, 6]  = raw[0] / 100.0
        struct_scalar[i, :, 7]  = raw[1] / 100.0
        struct_scalar[i, :, 8]  = raw[2] / 50.0
        struct_scalar[i, :, 9]  = raw[3] / 100.0
        struct_scalar[i, :, 10] = raw[4]  # mfe_frequency as-is
        struct_scalar[i, :, 11] = raw[5] / 50.0

    return struct_scalar, scalars_raw


def pearson_corr(x, y):
    """Pearson correlation between two arrays."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def main():
    device = DEVICE
    print(f'Device: {device}')

    # 1. Generate sequences
    print(f'\n=== Generating {N_SEQUENCES} sequences from pretrained model ===')
    seq_onehot = generate_sequences(N_SEQUENCES, device)
    print(f'  Shape: {seq_onehot.shape}')

    # Decode to RNA
    rna_seqs = sequences_to_rna(seq_onehot)
    print(f'  Example: {rna_seqs[0][:50]}...')

    # 2. Run KernaFold
    print('\n=== Running KernaFold ===')
    kernafold, kf_variant = load_kernafold(device)
    kf_mfe_structs = []
    kf_cent_structs = []
    kf_scalars_raw = []

    bs = 16
    for i in range(0, N_SEQUENCES, bs):
        batch = seq_onehot[i:i+bs].to(device)
        with torch.no_grad():
            if kf_variant == 'bpp':
                mask = kernafold._compute_valid_pair_mask(batch)
                out = kernafold(batch, mask)
            else:
                out = kernafold(batch)

        # Structure argmax
        mfe_idx = out['mfe_logits'].argmax(dim=-1).cpu().numpy()  # [B, L]
        cent_idx = out['centroid_logits'].argmax(dim=-1).cpu().numpy()
        kf_mfe_structs.append(mfe_idx)
        kf_cent_structs.append(cent_idx)

        # Scalars (denormalized to raw units)
        raw = kernafold.denormalize_scalars(out['scalars_normed']).cpu().numpy()
        kf_scalars_raw.append(raw)

    kf_mfe_structs = np.concatenate(kf_mfe_structs, axis=0)
    kf_cent_structs = np.concatenate(kf_cent_structs, axis=0)
    kf_scalars_raw = np.concatenate(kf_scalars_raw, axis=0)
    print(f'  KernaFold done: {kf_mfe_structs.shape}, scalars {kf_scalars_raw.shape}')

    # 3. Run Vienna RNA API
    print(f'\n=== Running Vienna RNA API ({N_SEQUENCES} seqs) ===')
    t0 = time.time()
    vienna_results = call_vienna_batch(rna_seqs)
    elapsed = time.time() - t0
    n_ok = sum(1 for r in vienna_results if r is not None)
    print(f'  Vienna done: {n_ok}/{N_SEQUENCES} ok, {elapsed:.1f}s')

    vienna_struct_scalar, vienna_scalars_raw = encode_vienna_features(vienna_results)

    # Convert Vienna structures to index arrays for comparison
    vienna_mfe_idx = np.zeros((N_SEQUENCES, 197), dtype=np.int64)
    vienna_cent_idx = np.zeros((N_SEQUENCES, 197), dtype=np.int64)
    for i, result in enumerate(vienna_results):
        if result is None:
            continue
        for j, ch in enumerate(result['mfe']['structure'][:197]):
            vienna_mfe_idx[i, j] = _STRUCT_CHAR_TO_IDX.get(ch, 0)
        for j, ch in enumerate(result['centroid']['structure'][:197]):
            vienna_cent_idx[i, j] = _STRUCT_CHAR_TO_IDX.get(ch, 0)

    # Filter out failed Vienna calls
    valid = [i for i, r in enumerate(vienna_results) if r is not None]
    if len(valid) < N_SEQUENCES:
        print(f'  WARNING: {N_SEQUENCES - len(valid)} Vienna calls failed, using {len(valid)} valid')

    # =====================================================================
    # 4. Compare structures
    # =====================================================================
    print('\n' + '=' * 70)
    print('KERNAFOLD vs VIENNA COMPARISON')
    print('=' * 70)

    # Per-position accuracy
    mfe_acc = np.mean(kf_mfe_structs[valid] == vienna_mfe_idx[valid])
    cent_acc = np.mean(kf_cent_structs[valid] == vienna_cent_idx[valid])
    print(f'\n--- Structure Accuracy (per-position) ---')
    print(f'  MFE structure:      {mfe_acc:.4f} ({mfe_acc*100:.1f}%)')
    print(f'  Centroid structure:  {cent_acc:.4f} ({cent_acc*100:.1f}%)')

    # Per-sequence structure accuracy
    per_seq_mfe = np.mean(kf_mfe_structs[valid] == vienna_mfe_idx[valid], axis=1)
    per_seq_cent = np.mean(kf_cent_structs[valid] == vienna_cent_idx[valid], axis=1)
    print(f'  MFE per-seq:        {np.mean(per_seq_mfe):.4f} +/- {np.std(per_seq_mfe):.4f}')
    print(f'  Centroid per-seq:   {np.mean(per_seq_cent):.4f} +/- {np.std(per_seq_cent):.4f}')

    # Breakdown by structure type
    for struct_name, kf_s, v_s in [('MFE', kf_mfe_structs, vienna_mfe_idx),
                                    ('Centroid', kf_cent_structs, vienna_cent_idx)]:
        kf_flat = kf_s[valid].flatten()
        v_flat = v_s[valid].flatten()
        for label, idx in [('unpaired (.)', 0), ('opening (()', 1), ('closing ())', 2)]:
            mask = v_flat == idx
            if mask.sum() > 0:
                acc = np.mean(kf_flat[mask] == idx)
                print(f'  {struct_name} {label}: {acc:.4f} ({mask.sum()} positions)')

    # =====================================================================
    # 5. Compare scalars
    # =====================================================================
    print(f'\n--- Scalar Comparison ---')
    print(f'  {"Scalar":<22s} {"Pearson":>8s} {"MAE":>8s} {"Vienna mean":>12s} {"KF mean":>12s}')
    print(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*12} {"-"*12}')
    for j, name in enumerate(_SCALAR_NAMES):
        v = vienna_scalars_raw[valid, j]
        k = kf_scalars_raw[valid, j]
        r = pearson_corr(v, k)
        mae = np.mean(np.abs(v - k))
        print(f'  {name:<22s} {r:8.4f} {mae:8.4f} {np.mean(v):12.4f} {np.mean(k):12.4f}')

    # =====================================================================
    # 6. Compare teacher rewards (Vienna-based vs KernaFold-based features)
    # =====================================================================
    print(f'\n--- Teacher Reward Comparison ---')
    teacher = load_teacher(device)

    L = 197
    seq_np = seq_onehot.cpu().numpy().astype(np.float32)

    # Build 16-ch from Vienna
    vienna_16ch = np.concatenate([seq_np, vienna_struct_scalar], axis=-1)  # [N, L, 16]
    vienna_16ch_t = torch.from_numpy(vienna_16ch).float().to(device)

    # Build 16-ch from KernaFold
    kf_struct_scalar = np.zeros((N_SEQUENCES, L, 12), dtype=np.float32)
    # MFE structure one-hot
    for i in range(N_SEQUENCES):
        for j in range(L):
            kf_struct_scalar[i, j, kf_mfe_structs[i, j]] = 1.0
            kf_struct_scalar[i, j, 3 + kf_cent_structs[i, j]] = 1.0
    # Scalars normalized and broadcast
    for i in range(N_SEQUENCES):
        kf_struct_scalar[i, :, 6]  = kf_scalars_raw[i, 0] / 100.0
        kf_struct_scalar[i, :, 7]  = kf_scalars_raw[i, 1] / 100.0
        kf_struct_scalar[i, :, 8]  = kf_scalars_raw[i, 2] / 50.0
        kf_struct_scalar[i, :, 9]  = kf_scalars_raw[i, 3] / 100.0
        kf_struct_scalar[i, :, 10] = kf_scalars_raw[i, 4]
        kf_struct_scalar[i, :, 11] = kf_scalars_raw[i, 5] / 50.0

    kf_16ch = np.concatenate([seq_np, kf_struct_scalar], axis=-1)
    kf_16ch_t = torch.from_numpy(kf_16ch).float().to(device)

    # Run teacher on both
    with torch.no_grad():
        reward_vienna = teacher(vienna_16ch_t).cpu().numpy()
        reward_kf = teacher(kf_16ch_t).cpu().numpy()

    r_rewards = pearson_corr(reward_vienna[valid], reward_kf[valid])
    mae_rewards = np.mean(np.abs(reward_vienna[valid] - reward_kf[valid]))
    print(f'  Teacher reward Pearson:  {r_rewards:.4f}')
    print(f'  Teacher reward MAE:      {mae_rewards:.4f}')
    print(f'  Vienna reward mean:      {np.mean(reward_vienna[valid]):.4f} +/- {np.std(reward_vienna[valid]):.4f}')
    print(f'  KernaFold reward mean:   {np.mean(reward_kf[valid]):.4f} +/- {np.std(reward_kf[valid]):.4f}')
    print(f'  Vienna reward range:     [{np.min(reward_vienna[valid]):.4f}, {np.max(reward_vienna[valid]):.4f}]')
    print(f'  KernaFold reward range:  [{np.min(reward_kf[valid]):.4f}, {np.max(reward_kf[valid]):.4f}]')

    # Scatter summary
    diff = reward_kf[valid] - reward_vienna[valid]
    print(f'  Reward diff (KF - Vienna): {np.mean(diff):.4f} +/- {np.std(diff):.4f}')
    print(f'  Max |diff|:                {np.max(np.abs(diff)):.4f}')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  Structure accuracy:  MFE={mfe_acc*100:.1f}%, Centroid={cent_acc*100:.1f}%')
    print(f'  Energy Pearson:      mfe={pearson_corr(vienna_scalars_raw[valid,0], kf_scalars_raw[valid,0]):.3f},'
          f' ens={pearson_corr(vienna_scalars_raw[valid,3], kf_scalars_raw[valid,3]):.3f}')
    print(f'  Teacher reward:      Pearson={r_rewards:.4f}, MAE={mae_rewards:.4f}')
    print(f'  Verdict: KernaFold {"GOOD" if r_rewards > 0.9 else "OK" if r_rewards > 0.7 else "POOR"}'
          f' proxy for Vienna (reward corr={r_rewards:.3f})')
    print('=' * 70)


if __name__ == '__main__':
    main()
