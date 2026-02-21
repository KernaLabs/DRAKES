# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DRAKES (Discrete Reward-Aligned Knowledge and Exploration for Sequences) is a fine-tuning method for reward optimization in discrete diffusion models, applied to DNA and protein sequence design. The method uses direct backpropagation with the softmax-gumbel trick for alignment.

This is the KernaLabs fork, extending the original work with a 3'UTR mRNA stability module.

Paper: "Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design" (ICLR 2025)

## Repository Structure

Three application domains, each with separate environments:

- `drakes_dna/` - Regulatory DNA sequence design (enhancer activity optimization)
- `drakes_protein/` - Protein sequence design (stability optimization via inverse folding)
- `drakes_narry_kim/` - 3'UTR mRNA stability optimization (viral tiles dataset)

## Environment Setup

### DNA Module (drakes_dna/)
```bash
conda create -n sedd python=3.9.18
conda activate sedd
bash drakes_dna/env.sh

# Install gReLU (version 1.0.2 required)
git clone https://github.com/Genentech/gReLU.git
cd gReLU && pip install .
```

### Protein Module (drakes_protein/)
```bash
conda env create -f drakes_protein/multiflow.yml
conda activate multiflow
pip install -e .
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# PyRosetta required for evaluation
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### 3'UTR Module (drakes_narry_kim/)
Uses the same conda environment as `drakes_dna/` (sedd), plus:
- RNABiMamba regressor for reward prediction
- KernaFold differentiable Vienna RNA surrogate (optional, at `/mnt/ssd1/code/kernafold/`)
- Vienna RNA API at `localhost:8000` for structural features

## Common Commands

### DNA Sequence Design
```bash
cd drakes_dna

# Pretrain diffusion model
python main_gosai.py

# Train reward oracle
python train_oracle.py

# Fine-tune with DRAKES (direct backpropagation)
python finetune_reward_bp.py --name <exp_name>
```

### Protein Sequence Design
```bash
cd drakes_protein

# Pretrain inverse folding model
python fmif/train_fmif.py --eval_every_n_epochs 100

# Train reward oracle
python protein_oracle/train_oracle.py --save_model_every_n_epochs=5 --wandb_name=test_ft

# Fine-tune with DRAKES
python fmif/finetune_reward_bp.py --wandb_name=<exp_name>

# Evaluate
cd fmif/scripts && bash ours.sh
```

### 3'UTR mRNA Stability
```bash
cd drakes_narry_kim

# Pretrain diffusion model
python main.py

# Fine-tune with DRAKES
python finetune_reward_bp.py
```

## Architecture

### Core Components

**DNA Module:**
- `diffusion_gosai_update.py` - Main Diffusion class (LightningModule) with sampling and fine-tuning methods
- `finetune_reward_bp.py` - DRAKES fine-tuning loop with KL regularization
- `oracle.py` - gReLU-based reward oracles for enhancer activity prediction
- `dataloader_gosai.py` - Gosai enhancer dataset loading
- `models/dnaconv.py` - CNN backbone for DNA sequences
- `models/dit.py` - DiT (Diffusion Transformer) backbone

**Protein Module:**
- `fmif/` - Flow-matching inverse folding implementation
- `fmif/finetune_reward_bp.py` - DRAKES fine-tuning with PyRosetta evaluation
- `protein_oracle/` - Protein stability reward oracle
- `multiflow/` - MultiFlow-based architecture (SE3 flows, IPA)
- `openfold/` - OpenFold components for structure prediction

**3'UTR Module:**
- `diffusion.py` - Adapted Diffusion class for 197bp viral tile sequences
- `finetune_reward_bp.py` - DRAKES fine-tuning with Vienna RNA features
- `kernafold_reward_wrapper.py` - Differentiable reward using KernaFold surrogate
- `vienna_reward_wrapper.py` - Reward wrapper using Vienna RNA API
- `models/dimamba.py` - DiMamba backbone
- `models/dit.py` - DiT backbone
- `models/dnaconv.py` - CNN backbone

### Key Fine-tuning Parameters
- `alpha` - KL divergence regularization weight
- `truncate_steps` - Number of diffusion steps to backprop through
- `gumbel_temp` - Temperature for Gumbel-softmax sampling
- `num_accum_steps` - Gradient accumulation steps

## Configuration

Uses Hydra for configuration management. Main config files:
- `drakes_dna/configs_gosai/config_gosai.yaml` - DNA training config
- `drakes_protein/multiflow/configs/` - Protein model configs
- `drakes_narry_kim/configs/config.yaml` - 3'UTR training config

## Data Requirements

Download data and pretrained weights from Dropbox (see README.md) and set `BASE_PATH` appropriately. Key paths to configure:
- `base_path` in `dataloader_gosai.py`, `oracle.py`, `finetune_reward_bp.py` (DNA)
- `base_path` in `fmif/finetune_reward_bp.py`, `fmif/train_fmif.py`, `protein_oracle/train_oracle.py` (Protein)

## Git Hooks

A pre-commit hook prevents committing files larger than 50MB (GitHub's LFS threshold). To install:
```bash
git config core.hooksPath .githooks
```

## Logging

Training logs to Weights & Biases by default. Use `--name debug` to disable W&B logging. Output saved to `<base_path>/<module>/reward_bp_results/`.
