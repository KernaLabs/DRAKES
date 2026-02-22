"""Centralized path resolution for DRAKES.

All large files (data, checkpoints, experiment outputs) live under a
single ``storage_root`` on a high-capacity drive.  Every script imports
this module instead of hardcoding paths.

Quick start::

    import drakes_paths as dp

    # DNA module
    dp.dna.gosai_csv          # Path to Gosai enhancer dataset
    dp.dna.pretrained_ckpt    # Pretrained diffusion model

    # 3'UTR module
    dp.narry_kim.viral_tiles_data   # Viral tiles dataset
    dp.narry_kim.regressor_ckpt_dir # RNABiMamba checkpoint dir

    # External code imports (avoids sys.path / importlib boilerplate)
    RNABiMamba = dp.import_rnabimamba()
    build_model = dp.import_kernafold_build()

Override ``storage_root`` without editing any file::

    DRAKES_STORAGE_ROOT=/opt/dlami/nvme/DRAKES python ...
"""

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

# ---------------------------------------------------------------------------
# Locate paths.yaml
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _REPO_ROOT / 'paths.yaml'

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing {_CONFIG_PATH}. "
        f"Copy paths.yaml.example to paths.yaml and set storage_root."
    )

with open(_CONFIG_PATH) as _f:
    _raw = yaml.safe_load(_f)

# Environment variable override takes precedence
_storage_root = Path(
    os.environ.get('DRAKES_STORAGE_ROOT', _raw['storage_root'])
)
_external = _raw.get('external', {})

# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------
storage_root = _storage_root
repo_root = _REPO_ROOT

# ---------------------------------------------------------------------------
# DNA module
# ---------------------------------------------------------------------------
dna = SimpleNamespace(
    # Data
    gosai_csv=_storage_root / 'dna' / 'data' / 'gosai_all.csv',
    gosai_dataset_gz=_storage_root / 'dna' / 'data' / 'dataset.csv.gz',
    atac_ckpt=_storage_root / 'dna' / 'data' / 'binary_atac_cell_lines.ckpt',
    ginkgo_pseudou_csv=_storage_root / 'dna' / 'data' / 'Ginkgo_pseudoU_3UTR_Dataset.csv',
    ncbi_cds_csv=_storage_root / 'dna' / 'data' / 'ncbi_cds_euk_w_hm.csv',
    hf_cache=_storage_root / 'dna' / 'cache',
    # Models
    pretrained_ckpt=_storage_root / 'dna' / 'models' / 'pretrained.ckpt',
    reward_oracle_ft=_storage_root / 'dna' / 'models' / 'reward_oracle_ft.ckpt',
    reward_oracle_eval=_storage_root / 'dna' / 'models' / 'reward_oracle_eval.ckpt',
    # Outputs
    outputs_dir=_storage_root / 'dna' / 'outputs',
    hydra_output_dir=_storage_root / 'dna' / 'outputs' / 'hydra_runs',
    reward_bp_results=_storage_root / 'dna' / 'outputs' / 'reward_bp_results',
    reward_bp_results_improved=_storage_root / 'dna' / 'outputs' / 'reward_bp_results_improved',
    reward_bp_results_random=_storage_root / 'dna' / 'outputs' / 'reward_bp_results_random',
)

# ---------------------------------------------------------------------------
# Narry Kim 3'UTR module
# ---------------------------------------------------------------------------
narry_kim = SimpleNamespace(
    # Data
    viral_tiles_data=_storage_root / 'narry_kim' / 'data' / 'viral_tiles_struct.jsonl.gz',
    genome_split_dir=_storage_root / 'narry_kim' / 'data' / 'rnet_genome_split_n1m_seed101',
    deseq_results=_storage_root / 'narry_kim' / 'data' / 'deseq2_results_fixed.csv',
    # Models
    regressor_ckpt_dir=_storage_root / 'narry_kim' / 'models' / 'regressor',
    kernafold_ckpt=_storage_root / 'narry_kim' / 'models' / 'kernafold_best.pt',
    # Experiments (pretrained diffusion checkpoints)
    experiments_dir=_storage_root / 'narry_kim' / 'experiments',
    experiments_cnn_large=_storage_root / 'narry_kim' / 'experiments_cnn_large',
    experiments_dit_large=_storage_root / 'narry_kim' / 'experiments_dit_large',
    experiments_dimamba_large=_storage_root / 'narry_kim' / 'experiments_dimamba_large',
    experiments_dit_large_highlr=_storage_root / 'narry_kim' / 'experiments_dit_large_highlr',
    experiments_dimamba_medium_highlr=_storage_root / 'narry_kim' / 'experiments_dimamba_medium_highlr',
    # Outputs
    outputs_dir=_storage_root / 'narry_kim' / 'outputs',
    hydra_output_dir=_storage_root / 'narry_kim' / 'outputs' / 'hydra_runs',
    reward_bp_results=_storage_root / 'narry_kim' / 'outputs' / 'reward_bp_results',
    distilled_oracle=_storage_root / 'narry_kim' / 'outputs' / 'distilled_oracle',
)

# ---------------------------------------------------------------------------
# External code dependency helpers
# ---------------------------------------------------------------------------
_NARRY_KIM_MODELS_DIR = _external.get(
    'narry_kim_models', '/mnt/ssd1/code/narry_kim_2025/models')
_KERNAFOLD_DIR = _external.get(
    'kernafold', '/mnt/ssd1/code/kernafold')


def import_rnabimamba():
    """Import RNABiMamba class via importlib (avoids model.py name clashes)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'narry_kim_model',
        os.path.join(_NARRY_KIM_MODELS_DIR, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RNABiMamba


def import_kernafold_build():
    """Import KernaFold's build_model via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'kernafold_model',
        os.path.join(_KERNAFOLD_DIR, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model
