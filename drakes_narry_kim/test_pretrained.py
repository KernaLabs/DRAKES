"""Rigorous test of pretrained diffusion model."""
import torch
import sys
import numpy as np
import math
from collections import Counter

sys.path.insert(0, '.')

import uuid as uuid_lib
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

def div_up(a, b):
    a, b = int(a), int(b)
    return str(a // b + (1 if a % b != 0 else 0))

for name, fn in [
    ('uuid', lambda: str(uuid_lib.uuid4())),
    ('cwd', lambda: '.'),
    ('device_count', lambda: str(torch.cuda.device_count())),
    ('eval', lambda x: str(eval(x))),
    ('div_up', div_up),
]:
    try:
        OmegaConf.register_new_resolver(name, fn)
    except:
        pass

import drakes_paths as dp_paths
try:
    OmegaConf.register_new_resolver('drakes_root', lambda: str(dp_paths.storage_root))
except:
    pass

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(dp_paths.repo_root / 'drakes_narry_kim' / 'configs'), version_base=None):
    args = compose(config_name='config')

import diffusion
model = diffusion.Diffusion(args)
ckpt = torch.load(str(dp_paths.narry_kim.experiments_dir / 'checkpoints' / 'best.ckpt'), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval().cuda()
print(f'Loaded best ckpt: epoch {ckpt["epoch"]}, step {ckpt["global_step"]}')

# ====== 1. DATA DISTRIBUTION ======
print('\n' + '='*60)
print('=== DATA DISTRIBUTION ===')
print('='*60)
import dataloader
full_ds = dataloader.NarryKimDataset()
n_total = len(full_ds)
n_val = int(n_total * 0.1)
n_train = n_total - n_val
gen = torch.Generator().manual_seed(42)
train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=gen)

real_seqs = []
for i in range(min(1000, len(val_ds))):
    item = val_ds[i]
    seq_tensor = item['seqs'] if isinstance(item, dict) else item
    seq = ''.join({0:'A',1:'C',2:'G',3:'T'}[x.item()] for x in seq_tensor)
    real_seqs.append(seq)

all_real = ''.join(real_seqs)
print('Real data nucleotide composition:')
for nt in 'ACGT':
    print(f'  {nt}: {all_real.count(nt)/len(all_real):.4f}')

# Per-position entropy
real_entropy_per_pos = []
for pos in range(197):
    counts = Counter(seq[pos] for seq in real_seqs)
    total = sum(counts.values())
    h = -sum((c/total) * math.log(c/total) for c in counts.values() if c > 0)
    real_entropy_per_pos.append(h)
mean_entropy = np.mean(real_entropy_per_pos)
print(f'Mean per-position entropy: {mean_entropy:.4f} (max possible: {math.log(4):.4f})')
print(f'Entropy ratio: {mean_entropy/math.log(4):.4f} (1.0 = completely uniform/random)')
print(f'This means: {"high entropy, data is nearly random" if mean_entropy/math.log(4) > 0.95 else "data has learnable structure"}')

# ====== 2. SAMPLE FROM MODEL ======
print('\n' + '='*60)
print('=== MODEL SAMPLES ===')
print('='*60)
with torch.no_grad():
    batch_size = 64
    x = model.mask_index * torch.ones(batch_size, 197, dtype=torch.int64).to(model.device)
    timesteps = torch.linspace(1, 1e-3, 129, device=model.device)
    dt = (1 - 1e-3) / 128
    for i in range(128):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
        x = model._ddpm_update(x, t, dt)

    gen_seqs = []
    for i in range(batch_size):
        seq = ''.join({0:'A',1:'C',2:'G',3:'T',4:'M'}[x[i,j].item()] for j in range(197))
        gen_seqs.append(seq)

all_gen = ''.join(gen_seqs)
print('Generated nucleotide composition:')
for nt in 'ACGTM':
    frac = all_gen.count(nt)/len(all_gen)
    print(f'  {nt}: {frac:.4f}')

print('\nSample generated sequences (first 5):')
for i in range(5):
    print(f'  {gen_seqs[i][:100]}...')

# ====== 3. K-MER COMPARISON ======
print('\n' + '='*60)
print('=== K-MER COMPARISON ===')
print('='*60)

def get_kmer_dist(seqs, k=3):
    counts = Counter()
    for s in seqs:
        s_clean = s.replace('M','')
        for j in range(len(s_clean)-k+1):
            counts[s_clean[j:j+k]] += 1
    total = sum(counts.values())
    return {kmer: c/total for kmer, c in counts.items()}

for k in [2, 3, 4]:
    real_km = get_kmer_dist(real_seqs, k)
    gen_km = get_kmer_dist(gen_seqs, k)
    all_kmers = set(real_km.keys()) | set(gen_km.keys())
    kl = 0
    for kmer in all_kmers:
        p = real_km.get(kmer, 1e-8)
        q = gen_km.get(kmer, 1e-8)
        kl += p * math.log(p / q)
    print(f'{k}-mer KL(real || generated): {kl:.6f}')

print('\nTop 10 real 3-mers:', sorted(get_kmer_dist(real_seqs, 3).items(), key=lambda x: -x[1])[:10])
print('Top 10 gen 3-mers:', sorted(get_kmer_dist(gen_seqs, 3).items(), key=lambda x: -x[1])[:10])

# ====== 4. DIRECT NLL EVALUATION ======
print('\n' + '='*60)
print('=== DIRECT NLL EVALUATION ===')
print('='*60)
from torch.utils.data import DataLoader
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
total_nll = 0
n_batches = 0
with torch.no_grad():
    for batch in val_loader:
        if isinstance(batch, dict):
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            batch = batch.cuda()
        loss = model.training_step(batch, 0)
        total_nll += loss.item()
        n_batches += 1
        if n_batches >= 20:
            break
avg_nll = total_nll / n_batches
print(f'Val NLL (20 batches): {avg_nll:.4f}')
print(f'Theoretical uniform NLL: {math.log(4):.4f}')
print(f'Improvement over uniform: {(math.log(4) - avg_nll)/math.log(4)*100:.2f}%')
print(f'Val perplexity: {math.exp(avg_nll):.4f}')
print(f'Uniform perplexity: {4.0:.4f}')

# ====== 5. COMPARE WITH ORIGINAL DRAKES DNA MODEL ======
print('\n' + '='*60)
print('=== CONTEXT: ORIGINAL DRAKES DNA NLL ===')
print('='*60)
print('For reference, the original DRAKES DNA model on Gosai enhancer data:')
print('  Typical NLL values are similar — DNA sequences have high entropy')
print('  because nucleotide usage is fairly uniform across positions.')
print()
print(f'Our data entropy ratio: {mean_entropy/math.log(4):.4f}')
if mean_entropy / math.log(4) > 0.95:
    print('>>> The data itself is nearly maximum entropy!')
    print('>>> This explains the small NLL improvement — there is little')
    print('>>> positional structure to learn beyond nucleotide frequencies.')
    print('>>> The model CAN still be useful for DRAKES fine-tuning.')
else:
    print('>>> Data has significant structure — model may be undertrained.')

# ====== 6. CHECK BACKBONE OUTPUT NORMS ======
print('\n' + '='*60)
print('=== BACKBONE DIAGNOSTICS ===')
print('='*60)
for name, p in model.named_parameters():
    if p.numel() > 100:
        print(f'  {name}: shape={list(p.shape)}, mean={p.mean().item():.6f}, std={p.std().item():.6f}, norm={p.norm().item():.4f}')

# ====== 7. SEQUENCE DIVERSITY ======
print('\n' + '='*60)
print('=== SEQUENCE DIVERSITY ===')
print('='*60)
unique_gen = len(set(gen_seqs))
print(f'Unique generated sequences: {unique_gen}/{len(gen_seqs)}')

# Pairwise hamming distances
dists = []
for i in range(min(30, len(gen_seqs))):
    for j in range(i+1, min(30, len(gen_seqs))):
        d = sum(a != b for a, b in zip(gen_seqs[i], gen_seqs[j]))
        dists.append(d)
print(f'Mean pairwise Hamming distance: {np.mean(dists):.1f} / 197')
print(f'Min: {np.min(dists)}, Max: {np.max(dists)}')

# Compare with real data diversity
real_dists = []
for i in range(min(30, len(real_seqs))):
    for j in range(i+1, min(30, len(real_seqs))):
        d = sum(a != b for a, b in zip(real_seqs[i], real_seqs[j]))
        real_dists.append(d)
print(f'Real data mean pairwise Hamming: {np.mean(real_dists):.1f} / 197')
