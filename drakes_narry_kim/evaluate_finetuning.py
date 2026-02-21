"""Evaluation framework for DRAKES fine-tuning of 3'UTR diffusion model.

Three evaluation levels:
1. Regressor Reward — predicted log2FC from RNABiMamba
2. Distribution Fidelity — k-mer KL, composition, diversity vs pretrained
3. Ground Truth Alignment — feature distributions vs high-confidence
   experimental data (DESeq2 padj < 0.1)

Reference features (from analyze_cngg_structure.py):
- GC content
- MFE energy, centroid distance, ensemble diversity, ensemble energy
- CNGG motif count and structural context (hairpin, pentaloop pos0)
"""

import gzip
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESEQ_PATH = (
    '/mnt/ssd1/code/narry_kim_2025/viral_mpra/processed_data/'
    'results/deseq2_results_fixed.csv'
)
VIRAL_TILES_PATH = (
    '/mnt/ssd1/code/v1_unified/data/processed/viral_tiles_struct.jsonl.gz'
)
VIENNA_API_URL = 'http://localhost:8000/jobs/analyze'
CNGG_PATTERN = re.compile(r'C[ACGU]GG')
LOOP_NAMES = {
    3: 'triloop', 4: 'tetraloop', 5: 'pentaloop',
    6: 'hexaloop', 7: 'heptaloop', 8: 'octaloop',
}


# ---------------------------------------------------------------------------
# Structure parsing (from analyze_cngg_structure.py)
# ---------------------------------------------------------------------------

def parse_dot_bracket(structure):
    """Convert dot-bracket to base pair list [(i, j), ...]."""
    pairs, stack = [], []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            pairs.append((stack.pop(), i))
    return pairs


def find_hairpin_loops(structure, pairs):
    """Find all hairpin loops: returns list of (loop_start, loop_end, loop_size, stem_length)."""
    pair_map = {i: j for i, j in pairs}
    pair_map.update({j: i for i, j in pairs})
    hairpins = []
    n = len(structure)
    i = 0
    while i < n:
        if structure[i] == '.':
            start = i
            while i < n and structure[i] == '.':
                i += 1
            end = i - 1
            if start > 0 and end < n - 1:
                left, right = start - 1, end + 1
                if structure[left] == '(' and structure[right] == ')':
                    if pair_map.get(left) == right:
                        stem_len = 1
                        l, r = left - 1, right + 1
                        while l >= 0 and r < n and pair_map.get(l) == r:
                            stem_len += 1
                            l -= 1
                            r += 1
                        hairpins.append({
                            'loop_start': start, 'loop_end': end,
                            'loop_size': end - start + 1,
                            'stem_length': stem_len,
                        })
        else:
            i += 1
    return hairpins


def extract_cngg_features(sequence, structure):
    """Extract CNGG motif features for a sequence + structure pair.

    Args:
        sequence: RNA string (ACGU alphabet)
        structure: dot-bracket string

    Returns:
        dict with CNGG-related features
    """
    pairs = parse_dot_bracket(structure)
    hairpins = find_hairpin_loops(structure, pairs)

    cngg_matches = list(CNGG_PATTERN.finditer(sequence.upper()))
    num_cngg = len(cngg_matches)
    num_cngg_hairpin = 0
    has_penta_pos0 = False
    has_penta_pos0_short = False

    for match in cngg_matches:
        pos = match.start()
        for hp in hairpins:
            if hp['loop_start'] <= pos <= hp['loop_end']:
                num_cngg_hairpin += 1
                pos_in_loop = pos - hp['loop_start']
                if hp['loop_size'] == 5 and pos_in_loop == 0:
                    has_penta_pos0 = True
                    if hp['stem_length'] <= 3:
                        has_penta_pos0_short = True
                break

    return {
        'num_cngg': num_cngg,
        'num_cngg_hairpin': num_cngg_hairpin,
        'has_penta_pos0': has_penta_pos0,
        'has_penta_pos0_short': has_penta_pos0_short,
    }


# ---------------------------------------------------------------------------
# Vienna API
# ---------------------------------------------------------------------------

def fold_sequence(sequence, api_url=VIENNA_API_URL):
    """Fold a single sequence via Vienna RNA API with retries."""
    import time as _time
    for attempt in range(3):
        try:
            resp = requests.post(
                api_url,
                json={'sequence': sequence, 'use_n1m': True},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < 2:
                _time.sleep(5 * (attempt + 1))
    # Return None on total failure
    return None


def fold_batch(sequences, api_url=VIENNA_API_URL, workers=10):
    """Fold sequences in parallel."""
    results = [None] * len(sequences)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fold_sequence, seq, api_url): i
            for i, seq in enumerate(sequences)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results


# ---------------------------------------------------------------------------
# Feature extraction from Vienna results
# ---------------------------------------------------------------------------

def extract_all_features(sequence, vienna_result):
    """Extract all evaluation features from a sequence + Vienna result.

    Returns dict with: gc_content, mfe_energy, centroid_distance,
    ensemble_diversity, ensemble_energy, centroid_energy, mfe_frequency,
    num_cngg, num_cngg_hairpin, has_penta_pos0, has_penta_pos0_short
    """
    gc = (sequence.count('G') + sequence.count('C')) / len(sequence)
    mfe_energy = vienna_result['mfe']['energy']
    mfe_struct = vienna_result['mfe']['structure']
    centroid_distance = vienna_result['centroid']['distance']
    centroid_energy = vienna_result['centroid']['energy']
    ensemble_energy = vienna_result['thermodynamics']['ensemble_energy']
    mfe_frequency = vienna_result['thermodynamics']['mfe_frequency']
    ensemble_diversity = vienna_result['thermodynamics']['ensemble_diversity']

    cngg = extract_cngg_features(sequence, mfe_struct)

    return {
        'gc_content': gc,
        'mfe_energy': mfe_energy,
        'centroid_distance': centroid_distance,
        'ensemble_diversity': ensemble_diversity,
        'ensemble_energy': ensemble_energy,
        'centroid_energy': centroid_energy,
        'mfe_frequency': mfe_frequency,
        **cngg,
    }


# ---------------------------------------------------------------------------
# KL divergence for 1D distributions
# ---------------------------------------------------------------------------

def kl_divergence_histogram(p_values, q_values, n_bins=50, eps=1e-8):
    """KL(P || Q) estimated via histogram binning.

    Args:
        p_values: samples from distribution P (reference)
        q_values: samples from distribution Q (generated)
        n_bins: number of histogram bins
        eps: smoothing constant

    Returns:
        KL divergence (float)
    """
    all_vals = np.concatenate([p_values, q_values])
    bins = np.linspace(all_vals.min() - 1e-6, all_vals.max() + 1e-6, n_bins + 1)
    p_hist, _ = np.histogram(p_values, bins=bins, density=True)
    q_hist, _ = np.histogram(q_values, bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


# ---------------------------------------------------------------------------
# k-mer distribution utilities
# ---------------------------------------------------------------------------

def get_kmer_dist(sequences, k=3):
    """Compute k-mer frequency distribution from sequences."""
    counts = Counter()
    for seq in sequences:
        for j in range(len(seq) - k + 1):
            counts[seq[j:j + k]] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {kmer: c / total for kmer, c in counts.items()}


def kmer_kl(real_dist, gen_dist):
    """KL(real || gen) on k-mer distributions."""
    all_kmers = set(real_dist.keys()) | set(gen_dist.keys())
    kl = 0.0
    for kmer in all_kmers:
        p = real_dist.get(kmer, 1e-8)
        q = gen_dist.get(kmer, 1e-8)
        kl += p * math.log(p / q)
    return kl


def get_positional_kmer_dists(sequences, k=3):
    """Compute per-position k-mer frequency distributions.

    For each position i in [0, L-k], counts which k-mer appears at that
    position across all sequences and normalizes to a distribution.

    Returns: list of dicts, one per position. Each dict maps kmer -> freq.
    """
    if not sequences:
        return []
    L = len(sequences[0])
    n_positions = L - k + 1
    positional_counts = [Counter() for _ in range(n_positions)]
    for seq in sequences:
        for i in range(n_positions):
            positional_counts[i][seq[i:i + k]] += 1
    n_seqs = len(sequences)
    return [{kmer: c / n_seqs for kmer, c in pos.items()}
            for pos in positional_counts]


def positional_kmer_kl(ref_pos_dists, gen_pos_dists):
    """Compute per-position KL(ref || gen) and return (mean, per-position array).

    Args:
        ref_pos_dists: list of dicts from get_positional_kmer_dists (reference)
        gen_pos_dists: list of dicts from get_positional_kmer_dists (generated)

    Returns:
        (mean_kl, kl_per_position) where kl_per_position is a list of floats.
    """
    n = min(len(ref_pos_dists), len(gen_pos_dists))
    kl_per_pos = []
    for i in range(n):
        all_kmers = set(ref_pos_dists[i].keys()) | set(gen_pos_dists[i].keys())
        kl = 0.0
        for kmer in all_kmers:
            p = ref_pos_dists[i].get(kmer, 1e-8)
            q = gen_pos_dists[i].get(kmer, 1e-8)
            kl += p * math.log(p / q)
        kl_per_pos.append(kl)
    return float(np.mean(kl_per_pos)), kl_per_pos


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class FinetuneEvaluator:
    """Comprehensive evaluation for DRAKES fine-tuning.

    Loads reference data once at init, then evaluate() can be called
    repeatedly during training.
    """

    def __init__(
        self,
        deseq_path=DESEQ_PATH,
        viral_tiles_path=VIRAL_TILES_PATH,
        vienna_api_url=VIENNA_API_URL,
        padj_threshold=0.1,
    ):
        self.vienna_api_url = vienna_api_url

        # Load DESeq2 high-confidence set
        deseq = pd.read_csv(deseq_path)
        hc = deseq[deseq['padj'] < padj_threshold].copy()
        self.ref_stable = hc[hc['log2FoldChange'] > 0].copy()
        self.ref_unstable = hc[hc['log2FoldChange'] < 0].copy()
        self.ref_all_hc = hc.copy()

        # Load viral tiles for matching
        vt_by_seq = {}
        with gzip.open(viral_tiles_path, 'rt') as f:
            for line in f:
                rec = json.loads(line)
                vt_by_seq[rec['sequence'].upper()] = rec

        # Match DESeq2 to viral tiles and extract features
        ref_features = []
        for _, row in hc.iterrows():
            seq = row['trimmed_sequence'].upper()
            # DNA to RNA for matching
            vt = vt_by_seq.get(seq) or vt_by_seq.get(seq.replace('U', 'T'))
            if vt is None:
                continue
            feat = {
                'log2fc': row['log2FoldChange'],
                'padj': row['padj'],
                'gc_content': (seq.count('G') + seq.count('C')) / len(seq),
                'mfe_energy': vt['mfe_energy'],
                'centroid_distance': vt['centroid_distance'],
                'ensemble_diversity': vt['ensemble_diversity'],
                'ensemble_energy': vt['ensemble_energy'],
                'centroid_energy': vt['centroid_energy'],
                'mfe_frequency': vt['mfe_frequency'],
            }
            cngg = extract_cngg_features(
                seq.replace('T', 'U'), vt['mfe_structure'])
            feat.update(cngg)
            ref_features.append(feat)

        self.ref_df = pd.DataFrame(ref_features)
        self.ref_stable_df = self.ref_df[self.ref_df['log2fc'] > 0]
        self.ref_unstable_df = self.ref_df[self.ref_df['log2fc'] < 0]

        # Pre-compute reference k-mer distributions (from all HC sequences)
        hc_seqs = [row['trimmed_sequence'].upper() for _, row in hc.iterrows()]
        self.ref_kmer_dists = {
            k: get_kmer_dist(hc_seqs, k) for k in [2, 3, 4]
        }
        self.ref_positional_kmer_dists = {
            k: get_positional_kmer_dists(hc_seqs, k) for k in [2, 3, 4]
        }

        print(f'FinetuneEvaluator loaded: {len(self.ref_df)} matched HC samples '
              f'({len(self.ref_stable_df)} stable, {len(self.ref_unstable_df)} unstable)')

    def evaluate(self, generated_sequences, predicted_rewards=None):
        """Run full evaluation on generated sequences.

        Args:
            generated_sequences: list of DNA strings (ACGT alphabet)
            predicted_rewards: optional np.ndarray of regressor predictions

        Returns:
            dict of all metrics
        """
        n = len(generated_sequences)
        metrics = {'n_generated': n}

        # ===== Level 1: Regressor Reward =====
        if predicted_rewards is not None:
            metrics['reward_mean'] = float(np.mean(predicted_rewards))
            metrics['reward_std'] = float(np.std(predicted_rewards))
            metrics['reward_median'] = float(np.median(predicted_rewards))
            metrics['reward_max'] = float(np.max(predicted_rewards))
            metrics['reward_min'] = float(np.min(predicted_rewards))
            # Compare to ref distributions
            ref_log2fc = self.ref_df['log2fc'].values
            metrics['reward_kl_vs_ref_all'] = kl_divergence_histogram(
                ref_log2fc, predicted_rewards)
            stable_log2fc = self.ref_stable_df['log2fc'].values
            if len(stable_log2fc) > 5:
                metrics['reward_kl_vs_ref_stable'] = kl_divergence_histogram(
                    stable_log2fc, predicted_rewards)

        # ===== Level 2: Distribution Fidelity =====
        # Nucleotide composition
        all_chars = ''.join(generated_sequences)
        for nt in 'ACGT':
            metrics[f'comp_{nt}'] = all_chars.count(nt) / len(all_chars) if all_chars else 0

        metrics['gc_content_mean'] = float(np.mean([
            (s.count('G') + s.count('C')) / len(s) for s in generated_sequences
        ]))

        # k-mer KL (global)
        for k in [2, 3, 4]:
            gen_km = get_kmer_dist(generated_sequences, k)
            metrics[f'kmer_{k}_kl'] = kmer_kl(self.ref_kmer_dists[k], gen_km)

        # k-mer KL (positional)
        for k in [2, 3, 4]:
            gen_pos = get_positional_kmer_dists(generated_sequences, k)
            mean_kl, kl_per_pos = positional_kmer_kl(
                self.ref_positional_kmer_dists[k], gen_pos)
            metrics[f'kmer_{k}_pos_kl_mean'] = mean_kl
            metrics[f'kmer_{k}_pos_kl_max'] = float(np.max(kl_per_pos))
            metrics[f'kmer_{k}_pos_kl_array'] = kl_per_pos

        # Sequence diversity
        metrics['unique_seqs'] = len(set(generated_sequences))
        if n >= 2:
            dists = []
            sample_n = min(n, 50)
            for i in range(sample_n):
                for j in range(i + 1, sample_n):
                    d = sum(a != b for a, b in zip(
                        generated_sequences[i], generated_sequences[j]))
                    dists.append(d)
            metrics['hamming_mean'] = float(np.mean(dists))
            metrics['hamming_min'] = int(np.min(dists))

        # ===== Level 3: Ground Truth Alignment =====
        # Fold generated sequences via Vienna API
        rna_seqs = [s.replace('T', 'U') for s in generated_sequences]
        vienna_results = fold_batch(rna_seqs, self.vienna_api_url)

        gen_features = []
        for seq, vr in zip(rna_seqs, vienna_results):
            if vr is not None:
                gen_features.append(extract_all_features(seq, vr))
        gen_df = pd.DataFrame(gen_features)

        if len(gen_df) > 0:
            # Compare continuous features to stable reference
            continuous_features = [
                'gc_content', 'mfe_energy', 'centroid_distance',
                'ensemble_diversity', 'ensemble_energy', 'mfe_frequency',
            ]
            for feat in continuous_features:
                gen_vals = gen_df[feat].dropna().values
                ref_vals = self.ref_stable_df[feat].dropna().values
                if len(gen_vals) > 5 and len(ref_vals) > 5:
                    metrics[f'{feat}_gen_mean'] = float(np.mean(gen_vals))
                    metrics[f'{feat}_ref_stable_mean'] = float(np.mean(ref_vals))
                    metrics[f'{feat}_kl_vs_stable'] = kl_divergence_histogram(
                        ref_vals, gen_vals)

            # CNGG features (binary — compare fractions)
            for feat in ['has_penta_pos0', 'has_penta_pos0_short']:
                gen_frac = gen_df[feat].mean()
                ref_frac = self.ref_stable_df[feat].mean()
                metrics[f'{feat}_gen_frac'] = float(gen_frac)
                metrics[f'{feat}_ref_stable_frac'] = float(ref_frac)

            metrics['num_cngg_gen_mean'] = float(gen_df['num_cngg'].mean())
            metrics['num_cngg_ref_stable_mean'] = float(
                self.ref_stable_df['num_cngg'].mean())

        return metrics

    def print_report(self, metrics):
        """Pretty-print evaluation metrics."""
        print('\n' + '=' * 70)
        print('DRAKES FINE-TUNING EVALUATION')
        print('=' * 70)

        print(f'\nGenerated: {metrics["n_generated"]} sequences')

        # Level 1
        if 'reward_mean' in metrics:
            print(f'\n--- Level 1: Regressor Reward ---')
            print(f'  Predicted log2FC: {metrics["reward_mean"]:.4f} +/- {metrics["reward_std"]:.4f}')
            print(f'  Range: [{metrics["reward_min"]:.4f}, {metrics["reward_max"]:.4f}]')
            if 'reward_kl_vs_ref_all' in metrics:
                print(f'  KL vs ref (all HC):   {metrics["reward_kl_vs_ref_all"]:.4f}')
            if 'reward_kl_vs_ref_stable' in metrics:
                print(f'  KL vs ref (stable):   {metrics["reward_kl_vs_ref_stable"]:.4f}')

        # Level 2
        print(f'\n--- Level 2: Distribution Fidelity ---')
        print(f'  Composition: A={metrics.get("comp_A", 0):.3f} '
              f'C={metrics.get("comp_C", 0):.3f} '
              f'G={metrics.get("comp_G", 0):.3f} '
              f'T={metrics.get("comp_T", 0):.3f}')
        print(f'  GC content: {metrics.get("gc_content_mean", 0):.3f}')
        for k in [2, 3, 4]:
            pos_mean = metrics.get(f'kmer_{k}_pos_kl_mean', 0)
            pos_max = metrics.get(f'kmer_{k}_pos_kl_max', 0)
            print(f'  {k}-mer KL: {metrics.get(f"kmer_{k}_kl", 0):.6f}'
                  f'  (positional: mean={pos_mean:.6f}, max={pos_max:.6f})')
        if 'hamming_mean' in metrics:
            print(f'  Hamming mean: {metrics["hamming_mean"]:.1f}/197, '
                  f'min: {metrics.get("hamming_min", 0)}')
        print(f'  Unique seqs: {metrics.get("unique_seqs", 0)}/{metrics["n_generated"]}')

        # Level 3
        print(f'\n--- Level 3: Ground Truth Alignment ---')
        print(f'  {"Feature":<25} {"Generated":>12} {"Ref Stable":>12} {"KL":>10}')
        print(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*10}')
        for feat in ['gc_content', 'mfe_energy', 'centroid_distance',
                      'ensemble_diversity', 'ensemble_energy']:
            gen_key = f'{feat}_gen_mean'
            ref_key = f'{feat}_ref_stable_mean'
            kl_key = f'{feat}_kl_vs_stable'
            if gen_key in metrics:
                print(f'  {feat:<25} {metrics[gen_key]:>12.4f} '
                      f'{metrics[ref_key]:>12.4f} {metrics.get(kl_key, 0):>10.4f}')

        print(f'\n  CNGG Motifs:')
        print(f'    Count:          gen={metrics.get("num_cngg_gen_mean", 0):.2f}  '
              f'ref_stable={metrics.get("num_cngg_ref_stable_mean", 0):.2f}')
        print(f'    Penta pos0:     gen={metrics.get("has_penta_pos0_gen_frac", 0):.3f}  '
              f'ref_stable={metrics.get("has_penta_pos0_ref_stable_frac", 0):.3f}')
        print(f'    Penta pos0 short: gen={metrics.get("has_penta_pos0_short_gen_frac", 0):.3f}  '
              f'ref_stable={metrics.get("has_penta_pos0_short_ref_stable_frac", 0):.3f}')

        print('=' * 70)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    evaluator = FinetuneEvaluator()

    # Generate random DNA sequences as baseline test
    import random
    random.seed(42)
    test_seqs = []
    for _ in range(32):
        seq = ''.join(random.choice('ACGT') for _ in range(197))
        test_seqs.append(seq)

    print('\nEvaluating 32 random sequences...')
    metrics = evaluator.evaluate(test_seqs)
    evaluator.print_report(metrics)
