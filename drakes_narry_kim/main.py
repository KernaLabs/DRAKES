# =============================================================================
# Pretraining Entry Point for Narry Kim 3'UTR Diffusion Model
# =============================================================================
#
# Trains an absorbing-state discrete diffusion model (DiT backbone) on ~196k
# viral tile 3'UTR sequences (197bp) from viral_tiles_struct.jsonl.gz.
#
# The diffusion model learns to generate realistic 3'UTR sequences by
# progressively denoising from fully masked tokens. Once pretrained, it can
# be fine-tuned with DRAKES (finetune_reward_bp.py) to optimize for mRNA
# stability as predicted by the RNABiMamba regressor.
#
# Usage:
#   python main.py                           # Train with default config
#   python main.py trainer.max_steps=80000   # Override config via CLI
#   python main.py trainer.devices=1         # Single GPU
#
# Configuration:
#   configs/config.yaml      - Main training config
#   configs/model/tiny.yaml  - DiT architecture (512 hidden, 8 blocks)
#
# Vocab: {A=0, C=1, G=2, T=3, MASK=4} (5 tokens total)
# =============================================================================

import os
import datetime
import random
import string

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import wandb

import dataloader
import diffusion
import utils

# ---- Hydra custom resolvers ----
# uuid: generates a unique run ID for W&B and output directories
omegaconf.OmegaConf.register_new_resolver(
    "uuid",
    lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))
    + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
    use_cache=False)
# cwd: returns the current working directory
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
# device_count: returns the number of available CUDA devices
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
# eval: evaluates a Python expression (used for batch size calculations)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
# div_up: ceiling division (used for batch size / accumulation calculations)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
import drakes_paths as dp
omegaconf.OmegaConf.register_new_resolver('drakes_root', lambda: str(dp.storage_root), use_cache=True)


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints the full Hydra configuration tree using Rich formatting.

    Also saves the config tree to a text file in the checkpointing directory
    for reproducibility.

    Args:
        config: Hydra DictConfig with all training parameters.
        resolve: Whether to resolve variable interpolations before printing.
        save_cfg: Whether to save the config tree to disk.
    """
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, test_ds):
    """Prints a sample batch from each dataloader for sanity checking."""
    for dl_type, dl in [
        ('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
        if dl is None:
            continue
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('  Batch seqs.shape:', batch['seqs'].shape)
        print(f'  tokens: {dataloader.dna_detokenize(batch["seqs"][0])}')
        print(f'  ids: {batch["seqs"][0]}')


def _train(config, logger):
    """Main training loop.

    Sets up W&B logging, checkpoint resumption, Lightning callbacks,
    dataloaders, and the Diffusion model, then runs trainer.fit().

    Args:
        config: Hydra DictConfig with all training parameters.
        logger: Python logger instance.
    """
    logger.info('Starting Training.')

    # -- W&B logger setup --
    wandb_logger = None
    wandb_settings = wandb.Settings(
        base_url='https://api.wandb.ai'
    )
    if config.get('wandb', None) is not None and not config.debug_mode:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            settings=wandb_settings,
            **config.wandb)

    # -- Checkpoint resumption --
    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(
                config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # -- Lightning callbacks --
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    # -- Data --
    train_ds, valid_ds, test_ds = dataloader.get_dataloaders(config)

    # -- Model --
    model = diffusion.Diffusion(config, eval=True)

    # -- Trainer --
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)

    print('Start training...')
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Hydra entry point for pretraining.

    Seeds everything for reproducibility, prints the config, and
    launches training.
    """
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    assert config.mode == 'train'
    _train(config, logger)


if __name__ == '__main__':
    main()
