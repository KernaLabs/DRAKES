#!/usr/bin/env python3
"""
Distill 5-fold RNABiMamba ensemble (16-channel input) into a sequence-only
student model (4-channel input) for use with DRAKES.

Steps:
1. Load all 196k sequences from the jsonl, extract 16-channel features
2. Run 5-fold teacher ensemble → average predictions = soft labels
3. Train sequence-only student on (seq_oh → soft_label) with MSE loss
4. Evaluate on all 3309 labeled sequences against actual log2fc
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add teacher model path
sys.path.insert(0, str(Path("/mnt/ssd1/code/narry_kim_2025/models")))
from model import RNABiMamba

# ── Paths ──────────────────────────────────────────────────────────────────
TEACHER_DIR = Path(
    "/mnt/ssd1/code/narry_kim_2025/models/checkpoints/"
    "mamba_rnet_ablation_single_linear_head_lr1e-04_d256_L8_kfold5_genome"
)
DATA_FILE = Path(
    "/mnt/ssd1/code/v1_unified/data/processed/viral_tiles_struct.jsonl.gz"
)
GENOME_SPLIT_DIR = Path(
    "/mnt/ssd1/code/narry_kim_2025/models/data/rnet_genome_split_n1m_seed101"
)
OUT_DIR = Path("/mnt/ssd1/code/DRAKES/drakes_narry_kim/distilled_oracle")

# ── Teacher config ─────────────────────────────────────────────────────────
TEACHER_D_INPUT = 16
TEACHER_D_MODEL = 256
TEACHER_N_LAYERS = 8
TEACHER_DROPOUT = 0.1
NUM_FOLDS = 5
FEATURE_MASK = [0, 1, 2, 3, 19, 20, 21, 23, 24, 25, 29, 30, 31, 32, 33, 34]


# ── Data loading ──────────────────────────────────────────────────────────

def load_full_dataset():
    """
    Load all 196k sequences from the jsonl using vectorized encoding.

    Returns:
        X_16: [N, 197, 16] teacher input features
        X_4:  [N, 197, 4]  student input features (seq one-hot)
        y:    [N]           actual log2fc (0.0 if unlabeled)
        has_label: [N]      bool mask for labeled sequences
    """
    seqs = []
    mfe_structs = []
    cent_structs = []
    scalars = []
    y_list = []
    has_label_list = []

    with gzip.open(DATA_FILE, "rt") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            seqs.append(row["sequence"])
            mfe_structs.append(row["mfe_structure"])
            cent_structs.append(row["centroid_structure"])
            scalars.append([
                row["mfe_energy"] / 100,
                row["centroid_energy"] / 100,
                row["centroid_distance"] / 50,
                row["ensemble_energy"] / 100,
                row["mfe_frequency"],
                row["ensemble_diversity"] / 50,
            ])
            labeled = bool(row.get("has_label", False))
            y_list.append(float(row["log2fc"]) if labeled else 0.0)
            has_label_list.append(labeled)

            if (i + 1) % 50000 == 0:
                print(f"  Parsed {i + 1} lines...")

    N = len(seqs)
    L = len(seqs[0])
    print(f"  Parsed {N} sequences, encoding...")

    # Vectorized one-hot encoding
    seq_idx = np.array([[{"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}[c] for c in s] for s in seqs], dtype=np.int64)
    mfe_idx = np.array([[{".": 0, "(": 1, ")": 2}[c] for c in s] for s in mfe_structs], dtype=np.int64)
    cent_idx = np.array([[{".": 0, "(": 1, ")": 2}[c] for c in s] for s in cent_structs], dtype=np.int64)

    seq_oh = np.zeros((N, L, 4), dtype=np.float32)
    np.put_along_axis(seq_oh, seq_idx[:, :, None], 1.0, axis=2)

    mfe_oh = np.zeros((N, L, 3), dtype=np.float32)
    np.put_along_axis(mfe_oh, mfe_idx[:, :, None], 1.0, axis=2)

    cent_oh = np.zeros((N, L, 3), dtype=np.float32)
    np.put_along_axis(cent_oh, cent_idx[:, :, None], 1.0, axis=2)

    scalar_arr = np.array(scalars, dtype=np.float32)  # [N, 6]
    scalar_broadcast = np.broadcast_to(scalar_arr[:, None, :], (N, L, 6)).copy()  # [N, L, 6]

    X_16 = torch.from_numpy(np.concatenate([seq_oh, mfe_oh, cent_oh, scalar_broadcast], axis=2))
    X_4 = torch.from_numpy(seq_oh.copy())
    y = torch.tensor(y_list, dtype=torch.float32)
    has_label = torch.tensor(has_label_list, dtype=torch.bool)

    print(f"Loaded {N} sequences ({has_label.sum().item()} labeled)")
    return X_16, X_4, y, has_label


# ── Teacher ensemble ──────────────────────────────────────────────────────

def load_teachers(device: str = "cuda"):
    """Load 5-fold teacher models."""
    teachers = []
    for fold in range(NUM_FOLDS):
        model = RNABiMamba(
            d_input=TEACHER_D_INPUT,
            d_model=TEACHER_D_MODEL,
            n_layers=TEACHER_N_LAYERS,
            dropout=TEACHER_DROPOUT,
        )
        ckpt_path = TEACHER_DIR / f"fold_{fold}" / "best_model_sig.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
        print(f"  Loaded fold {fold}")
    return teachers


@torch.no_grad()
def generate_soft_labels(teachers, X_16, device="cuda", batch_size=1024):
    """Run ensemble on all data, return averaged predictions."""
    N = X_16.shape[0]
    all_preds = torch.zeros(NUM_FOLDS, N)

    for fold_idx, teacher in enumerate(teachers):
        preds = []
        for start in range(0, N, batch_size):
            batch = X_16[start : start + batch_size].to(device)
            pred = teacher(batch).cpu()
            preds.append(pred)
        all_preds[fold_idx] = torch.cat(preds)
        print(
            f"  Fold {fold_idx}: mean={all_preds[fold_idx].mean():.4f}, "
            f"std={all_preds[fold_idx].std():.4f}"
        )

    avg = all_preds.mean(dim=0)
    fold_disagreement = all_preds.std(dim=0).mean()
    print(
        f"  Ensemble: mean={avg.mean():.4f}, std={avg.std():.4f}, "
        f"avg inter-fold std={fold_disagreement:.4f}"
    )
    return avg


# ── Student training (DDP) ───────────────────────────────────────────────

def train_worker(rank, world_size, train_X, train_y, val_X, val_y,
                 student_config, lr, epochs, batch_size, resume_epoch):
    """Training worker for DDP."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    student = RNABiMamba(
        d_input=4,
        d_model=student_config["d_model"],
        n_layers=student_config["n_layers"],
        dropout=student_config["dropout"],
    ).to(device)

    # Resume from checkpoint if available
    ckpt_path = OUT_DIR / "student_checkpoint.pt"
    if resume_epoch > 0 and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        student.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print(f"  Resumed from epoch {resume_epoch}")

    student = DDP(student, device_ids=[rank])
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Student: d_model={student_config['d_model']}, "
              f"n_layers={student_config['n_layers']}, params={n_params:,}, "
              f"batch_size={batch_size}x{world_size}={batch_size*world_size}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Fast-forward scheduler if resuming
    for _ in range(resume_epoch):
        scheduler.step()

    criterion = nn.MSELoss()

    train_ds = TensorDataset(train_X, train_y)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True,
    )

    best_val_loss = float("inf")

    for epoch in range(resume_epoch, epochs):
        train_sampler.set_epoch(epoch)

        # ── Train ──
        student.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = student(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── Validate (rank 0 only) ──
        if rank == 0:
            student.eval()
            with torch.no_grad():
                val_pred = []
                for start in range(0, len(val_X), batch_size * 2):
                    batch = val_X[start : start + batch_size * 2].to(device)
                    val_pred.append(student.module(batch).cpu())
                val_pred = torch.cat(val_pred)
                val_loss = criterion(val_pred, val_y).item()
                pearson_teacher = torch.corrcoef(
                    torch.stack([val_pred, val_y])
                )[0, 1].item()

            train_loss = np.mean(train_losses)
            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = " *"
                # Save best checkpoint
                torch.save({
                    "model_state_dict": student.module.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "pearson_teacher": pearson_teacher,
                    "config": student_config,
                }, OUT_DIR / "student_best.pt")

            # Save latest checkpoint every epoch
            torch.save({
                "model_state_dict": student.module.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "pearson_teacher": pearson_teacher,
                "config": student_config,
            }, OUT_DIR / "student_checkpoint.pt")

            if epoch % 5 == 0 or improved:
                print(
                    f"  Epoch {epoch:3d}: train_mse={train_loss:.6f}  "
                    f"val_mse={val_loss:.6f}  "
                    f"pearson_teacher={pearson_teacher:.4f}"
                    f"{improved}"
                )

        dist.barrier()

    dist.destroy_process_group()


def train_student_ddp(train_X, train_y, val_X, val_y, student_config,
                      lr, epochs, batch_size, world_size, resume_epoch=0):
    """Launch DDP training across GPUs."""
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        train_worker,
        args=(world_size, train_X, train_y, val_X, val_y,
              student_config, lr, epochs, batch_size, resume_epoch),
        nprocs=world_size,
        join=True,
    )


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_all_labeled(student, teachers, device="cuda", batch_size=512):
    """Evaluate on all 3309 labeled sequences (train+test combined, since CV used all)."""
    train_data = torch.load(GENOME_SPLIT_DIR / "train.pt", weights_only=False)
    test_data = torch.load(GENOME_SPLIT_DIR / "test.pt", weights_only=False)

    # Combine train + test (CV used all of them)
    all_X_35 = torch.cat([train_data["X"], test_data["X"]], dim=0)
    all_y = torch.cat([train_data["y_log2fc"], test_data["y_log2fc"]], dim=0)
    all_padj = torch.cat([train_data["y_padj"], test_data["y_padj"]], dim=0)

    all_X_4 = all_X_35[:, :, :4]
    all_X_16 = all_X_35[:, :, FEATURE_MASK]

    # Teacher ensemble predictions
    teacher_preds = torch.zeros(NUM_FOLDS, len(all_y))
    for fold_idx, teacher in enumerate(teachers):
        for start in range(0, len(all_y), batch_size):
            batch = all_X_16[start:start+batch_size].to(device)
            teacher_preds[fold_idx, start:start+batch_size] = teacher(batch).cpu()
    teacher_avg = teacher_preds.mean(dim=0)

    # Student predictions
    student.eval()
    student_pred = []
    for start in range(0, len(all_X_4), batch_size):
        batch = all_X_4[start:start+batch_size].to(device)
        student_pred.append(student(batch).cpu())
    student_pred = torch.cat(student_pred)

    # Metrics on all sequences
    r_teacher = torch.corrcoef(torch.stack([teacher_avg, all_y]))[0, 1].item()
    r_student = torch.corrcoef(torch.stack([student_pred, all_y]))[0, 1].item()
    r_distill = torch.corrcoef(torch.stack([student_pred, teacher_avg]))[0, 1].item()

    print(f"\n=== All labeled sequences (n={len(all_y)}) ===")
    print(f"  Teacher Pearson vs true:    {r_teacher:.4f}")
    print(f"  Student Pearson vs true:    {r_student:.4f}")
    print(f"  Student Pearson vs teacher: {r_distill:.4f}")

    # Significant sequences (padj < 0.1)
    sig_mask = all_padj < 0.1
    if sig_mask.sum() > 10:
        r_teacher_sig = torch.corrcoef(
            torch.stack([teacher_avg[sig_mask], all_y[sig_mask]])
        )[0, 1].item()
        r_student_sig = torch.corrcoef(
            torch.stack([student_pred[sig_mask], all_y[sig_mask]])
        )[0, 1].item()
        print(f"\n=== Significant tiles (padj<0.1, n={sig_mask.sum().item()}) ===")
        print(f"  Teacher Pearson vs true:    {r_teacher_sig:.4f}")
        print(f"  Student Pearson vs true:    {r_student_sig:.4f}")

    return {
        "pearson_student_true": r_student,
        "pearson_teacher_true": r_teacher,
        "pearson_student_teacher": r_distill,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Per-GPU batch size")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument(
        "--skip_soft_labels", action="store_true",
        help="Load pre-computed soft labels instead of regenerating",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    soft_labels_path = OUT_DIR / "soft_labels.pt"
    data_cache_path = OUT_DIR / "data_cache.pt"

    # ── Step 1: Load data ──
    if data_cache_path.exists():
        print("Loading cached data tensors...")
        cache = torch.load(data_cache_path, weights_only=True)
        X_16, X_4, y_true, has_label = (
            cache["X_16"], cache["X_4"], cache["y"], cache["has_label"]
        )
        print(f"  {len(X_16)} sequences ({has_label.sum().item()} labeled)")
    else:
        print("Loading dataset from jsonl...")
        X_16, X_4, y_true, has_label = load_full_dataset()
        torch.save(
            {"X_16": X_16, "X_4": X_4, "y": y_true, "has_label": has_label},
            data_cache_path,
        )
        print(f"Cached data tensors to {data_cache_path}")

    # ── Step 2: Generate soft labels ──
    if args.skip_soft_labels and soft_labels_path.exists():
        print("\nLoading pre-computed soft labels...")
        soft_labels = torch.load(soft_labels_path, weights_only=True)
    else:
        print("\nLoading teacher ensemble...")
        teachers = load_teachers("cuda:0")
        print("Generating soft labels for all sequences...")
        soft_labels = generate_soft_labels(teachers, X_16, "cuda:0")
        torch.save(soft_labels, soft_labels_path)
        print(f"Saved soft labels to {soft_labels_path}")
        del teachers
        torch.cuda.empty_cache()

    # ── Step 3: Train/val split ──
    N = len(X_4)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_size = N // 10
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_X = X_4[train_idx]
    train_y = soft_labels[train_idx]
    val_X = X_4[val_idx]
    val_y = soft_labels[val_idx]

    print(f"\nTrain: {len(train_X)}, Val: {len(val_X)}")

    # Check for existing checkpoint to resume
    resume_epoch = 0
    ckpt_path = OUT_DIR / "student_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        resume_epoch = ckpt["epoch"] + 1
        print(f"Found checkpoint at epoch {ckpt['epoch']} "
              f"(val_mse={ckpt['val_loss']:.6f}, pearson={ckpt['pearson_teacher']:.4f})")
        if resume_epoch >= args.epochs:
            print("Training already complete, skipping to evaluation.")
        else:
            print(f"Resuming from epoch {resume_epoch}")

    # ── Step 4: Train student (DDP) ──
    if resume_epoch < args.epochs:
        student_config = {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        }
        print(f"\nTraining student on {args.gpus} GPUs...")
        train_student_ddp(
            train_X, train_y, val_X, val_y,
            student_config, args.lr, args.epochs, args.batch_size,
            world_size=args.gpus, resume_epoch=resume_epoch,
        )

    # ── Step 5: Evaluate ──
    print("\nLoading best student for evaluation...")
    best_path = OUT_DIR / "student_best.pt"
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    student = RNABiMamba(
        d_input=4,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    student.load_state_dict(best_ckpt["model_state_dict"])
    student.to("cuda:0")
    student.eval()
    print(f"  Best epoch: {best_ckpt['epoch']}, "
          f"val_mse={best_ckpt['val_loss']:.6f}, "
          f"pearson_teacher={best_ckpt['pearson_teacher']:.4f}")

    teachers = load_teachers("cuda:0")
    metrics = evaluate_on_all_labeled(student, teachers, "cuda:0")

    # Save final model
    save_path = OUT_DIR / "student_model.pt"
    torch.save({
        "model_state_dict": best_ckpt["model_state_dict"],
        "config": {
            "model": "mamba",
            "d_input": 4,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
        },
        "metrics": metrics,
        "best_epoch": best_ckpt["epoch"],
    }, save_path)
    print(f"\nSaved student model to {save_path}")

    # ── Gradient flow check ──
    print("\nGradient flow check...")
    x_test = torch.randn(2, 197, 4, requires_grad=True, device="cuda:0")
    out = student(x_test)
    out.sum().backward()
    print(f"  Input grad norm: {x_test.grad.norm().item():.6f}")
    print(f"  Gradients flow: {x_test.grad.abs().sum().item() > 0}")


if __name__ == "__main__":
    main()
