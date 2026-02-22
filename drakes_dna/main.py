import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
import drakes_paths as dp
omegaconf.OmegaConf.register_new_resolver('drakes_root', lambda: str(dp.storage_root), use_cache=True)


def _load_from_checkpoint(config, tokenizer):
    if 'hf' in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
    
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool): Whether to resolve reference fields of DictConfig.
        save_cfg (bool): Whether to save the configuration tree to a file.
    """
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
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


def _train(config, logger):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)

    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(
                config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    # Get data loaders
    train_ds, valid_ds, test_ds = dataloader.get_dataloaders(config)

    # Get tokenizer for DNA sequences 
    tokenizer = dataloader.get_tokenizer(config)

    # Initialize model
    model = diffusion.Diffusion(config, tokenizer)

    # Create trainer
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    
    print('Start training...')
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs', config_name='ginkgo_3utr')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    assert config.mode == 'train'
    _train(config, logger)


if __name__ == '__main__':
    main()