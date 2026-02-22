# =============================================================================
# DRAKES Fine-Tuning: Direct Reward Backpropagation for 3'UTR Stability
# =============================================================================
#
# Fine-tunes a pretrained absorbing-state diffusion model to generate 3'UTR
# sequences with high mRNA stability (log2FC) as predicted by the RNABiMamba
# regressor.
#
# Method (DRAKES - Discrete Reward-Aligned Knowledge and Exploration):
# -------------------------------------------------------------------
# 1. Sample sequences from the diffusion model using Gumbel-softmax (soft)
#    for the last `truncate_steps` steps of the reverse process.
# 2. Pass soft one-hot samples through ViennaRewardWrapper, which:
#    a. Applies straight-through estimation for sequence channels (grad flows)
#    b. Calls Vienna RNA API for structure features (no grad, detached)
#    c. Concatenates [seq(4) + struct(6) + scalar(6)] = 16 channels
#    d. Feeds into frozen RNABiMamba regressor → log2FC reward
# 3. Compute KL divergence between the fine-tuned and pretrained model
#    to prevent mode collapse.
# 4. Optimize: loss = -mean(reward) + alpha * KL
#
# Gradient Flow:
# ~~~~~~~~~~~~~~
#   Diffusion model → Gumbel-softmax → soft one-hot [B, 197, 4]
#       → straight-through → RNABiMamba → reward
#       ↑ gradients flow through soft one-hot back to diffusion model
#       ↑ Vienna features are detached (no grad through folding)
#       ↑ RNABiMamba weights are frozen (only input gradients propagate)
#
# Usage:
#   # Fine-tune with default settings
#   python finetune_reward_bp.py \
#       --checkpoint_path experiments/checkpoints/last.ckpt \
#       --name 3utr_drakes
#
#   # Fine-tune with custom hyperparameters
#   python finetune_reward_bp.py \
#       --checkpoint_path experiments/checkpoints/last.ckpt \
#       --regressor_checkpoint_dir /path/to/regressor/checkpoint \
#       --alpha 0.001 \
#       --truncate_steps 50 \
#       --batch_size 16 \
#       --num_epochs 500 \
#       --name 3utr_drakes_alpha001
#
# Key Hyperparameters:
#   --alpha             KL regularization weight (default: 0.001)
#   --truncate_steps    Diffusion steps with gradient (default: 50 of 128)
#   --batch_size        Samples per step (default: 16, limited by Vienna API)
#   --gumbel_temp       Gumbel-softmax temperature (default: 1.0)
#   --num_accum_steps   Gradient accumulation steps (default: 4)
#   --total_num_steps   Total diffusion steps during sampling (default: 128)
#
# Output:
#   Saves to reward_bp_results/<run_name>/:
#     - model_<epoch>.ckpt   Model checkpoint every save_every_n_epochs
#     - log.txt              Training metrics log
# =============================================================================

import argparse
import datetime
import os
import random
import string
import sys

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import diffusion as diffusion_module
from utils import set_seed, str2bool
from evaluate_finetuning import FinetuneEvaluator

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
import drakes_paths as dp
omegaconf.OmegaConf.register_new_resolver('drakes_root', lambda: str(dp.storage_root), use_cache=True)

# Default paths
_DEFAULT_REGRESSOR_CKPT = str(dp.narry_kim.regressor_ckpt_dir)
_DEFAULT_BASE_PATH = str(dp.narry_kim.outputs_dir)


def _generate_eval_sequences(model, num_seqs, num_steps, device):
    """Generate sequences from the model for evaluation (no grad)."""
    idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'M'}
    model.eval()
    sequences = []
    remaining = num_seqs
    while remaining > 0:
        bsz = min(remaining, 32)
        with torch.no_grad():
            x = model.mask_index * torch.ones(
                bsz, 197, dtype=torch.int64, device=device)
            timesteps = torch.linspace(1, 1e-3, num_steps + 1, device=device)
            dt = (1 - 1e-3) / num_steps
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                x = model._ddpm_update(x, t, dt)
        for i in range(bsz):
            seq = ''.join(idx_to_nt[x[i, j].item()] for j in range(197))
            sequences.append(seq.replace('M', 'A'))
        remaining -= bsz
    model.train()
    return sequences


def fine_tune(new_model, reward_wrapper, old_model, args, log_path, save_path,
              evaluator=None, eps=1e-5):
    """Main DRAKES fine-tuning loop.

    Iteratively samples from the diffusion model, computes rewards via the
    RNABiMamba regressor (with Vienna RNA features), and updates the model
    to maximize reward while staying close to the pretrained distribution.

    Args:
        new_model: Diffusion model being fine-tuned (on CUDA).
        reward_wrapper: ViennaRewardWrapper providing differentiable rewards.
        old_model: Frozen copy of the pretrained diffusion model for KL.
        args: Parsed command-line arguments.
        log_path: Path to the training log file.
        save_path: Directory for saving checkpoints.
        evaluator: Optional FinetuneEvaluator for periodic evaluation.
        eps: Small epsilon for numerical stability in timestep spacing.

    Returns:
        batch_losses: List of per-batch loss values across all epochs.
    """
    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    # Configure diffusion model for fine-tuning
    new_model.config.finetuning.truncate_steps = args.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    batch_losses = []
    batch_rewards = []

    for epoch_num in range(args.num_epochs):
        rewards = []
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_model.train()

        for _step in range(args.num_accum_steps):
            # ----- Step 1: Sample from diffusion model -----
            # _sample_finetune_gradient returns:
            #   sample: [B, L, 4] straight-through soft one-hot (mask channel stripped)
            #   last_x_list: list of intermediate states for KL computation
            #   condt_list: list of sigma_t conditioning values
            #   move_chance_t_list: list of move chance values
            #   copy_flag_list: list of copy flags
            sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = \
                new_model._sample_finetune_gradient(
                    num_steps=args.total_num_steps,
                    eval_sp_size=args.batch_size,
                    copy_flag_temp=args.copy_flag_temp,
                )  # sample: [B, 197, 4]

            # ----- Step 2: Compute reward via ViennaRewardWrapper -----
            # The wrapper handles:
            #   - Straight-through estimation for sequence channels
            #   - Vienna RNA API calls for structure/scalar features
            #   - RNABiMamba forward pass → log2FC prediction
            reward = reward_wrapper(sample)  # [B]

            # Log the detached reward (same sequences, no extra API calls)
            rewards.append(reward.detach().cpu().numpy())

            # ----- Step 3: Compute KL divergence -----
            # KL regularization keeps the fine-tuned model close to the
            # pretrained distribution, preventing mode collapse.
            total_kl = []
            kl_start = args.total_num_steps - args.truncate_steps if args.truncate_kl else 0
            kl_timesteps = list(range(kl_start, args.total_num_steps, args.kl_every_n))
            for random_t in kl_timesteps:
                last_x = last_x_list[random_t]  # [B, L, 5]
                condt = condt_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                copy_flag = copy_flag_list[random_t]  # [B, L, 1]

                # Forward pass through both models to get predicted x0 distributions
                log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]      # [B, L, 4]
                with torch.no_grad():
                    old_device = next(old_model.parameters()).device
                    log_p_x0_old = old_model.forward(
                        last_x.to(old_device), condt.to(old_device)
                    )[:, :, :-1].to(last_x.device)  # [B, L, 4]

                p_x0 = log_p_x0.exp()        # [B, L, 4]
                p_x0_old = log_p_x0_old.exp()  # [B, L, 4]

                # KL divergence weighted by copy flag and move chance
                kl_div = copy_flag * (
                    -p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)
                ) / move_chance_t[0, 0, 0]
                kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))  # [B]
                total_kl.append(kl_div)

            # ----- Step 4: Compute total loss -----
            # Linear warmup for alpha (KL weight) if configured
            if epoch_num < args.alpha_schedule_warmup:
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha

            kl_loss = torch.stack(total_kl, 1).sum(1).mean()
            reward_loss = -torch.mean(reward) * args.reward_scale
            loss = reward_loss + kl_loss * current_alpha
            loss = loss / args.num_accum_steps

            # ----- Step 5: Backward + optimizer step -----
            loss.backward()
            if (_step + 1) % args.num_accum_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            batch_losses.append(loss.cpu().detach().numpy())
            batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            reward_losses.append(reward_loss.cpu().detach().numpy())
            kl_losses.append(kl_loss.cpu().detach().numpy())

        # ----- Epoch logging -----
        rewards = np.array(rewards)
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        kl_losses = np.array(kl_losses)

        print(
            f"Epoch {epoch_num}  "
            f"Mean reward {np.mean(rewards):.4f}  "
            f"Mean grad norm {tot_grad_norm:.4f}  "
            f"Mean loss {np.mean(losses):.4f}  "
            f"Mean reward loss {np.mean(reward_losses):.4f}  "
            f"Mean kl loss {np.mean(kl_losses):.4f}"
        )
        if args.name != 'debug':
            wandb.log({
                "epoch": epoch_num,
                "mean_reward": np.mean(rewards),
                "mean_grad_norm": tot_grad_norm,
                "mean_loss": np.mean(losses),
                "mean_reward_loss": np.mean(reward_losses),
                "mean_kl_loss": np.mean(kl_losses),
            })
        with open(log_path, 'a') as f:
            f.write(
                f"Epoch {epoch_num} "
                f"Mean reward {np.mean(rewards):.4f} "
                f"Mean grad norm {tot_grad_norm:.4f} "
                f"Mean loss {np.mean(losses):.4f} "
                f"Mean reward loss {np.mean(reward_losses):.4f} "
                f"Mean kl loss {np.mean(kl_losses):.4f}\n"
            )

        # ----- Checkpoint saving -----
        if (epoch_num + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")

        # ----- Periodic evaluation -----
        if evaluator is not None and (epoch_num + 1) % args.eval_every_n_epochs == 0:
            print(f'\n--- Evaluation at epoch {epoch_num} ---')
            eval_seqs = _generate_eval_sequences(
                new_model, args.eval_batch_size, args.total_num_steps,
                next(new_model.parameters()).device)
            # Get regressor predictions for generated sequences
            with torch.no_grad():
                eval_oh = torch.zeros(
                    len(eval_seqs), 197, 4,
                    device=next(new_model.parameters()).device)
                nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                for i, seq in enumerate(eval_seqs):
                    for j, ch in enumerate(seq):
                        eval_oh[i, j, nt_to_idx[ch]] = 1.0
                eval_rewards = reward_wrapper(eval_oh).cpu().numpy()

            eval_metrics = evaluator.evaluate(eval_seqs, eval_rewards)
            evaluator.print_report(eval_metrics)

            if args.name != 'debug':
                wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()
                           if isinstance(v, (int, float))}, step=epoch_num)

    if args.name != 'debug':
        wandb.finish()

    return batch_losses


# =============================================================================
# CLI argument parsing
# =============================================================================
argparser = argparse.ArgumentParser(
    description='DRAKES fine-tuning for 3\'UTR stability optimization',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Paths
argparser.add_argument(
    '--checkpoint_path', type=str, required=True,
    help='Path to pretrained diffusion model checkpoint (.ckpt)')
argparser.add_argument(
    '--regressor_checkpoint_dir', type=str,
    default=_DEFAULT_REGRESSOR_CKPT,
    help='Path to RNABiMamba regressor checkpoint directory')
argparser.add_argument(
    '--distilled_oracle_path', type=str, default=None,
    help='Path to distilled student model (.pt). If provided, uses fully '
         'differentiable sequence-only oracle instead of ViennaRewardWrapper.')
argparser.add_argument(
    '--kernafold_checkpoint_path', type=str, default=None,
    help='Path to KernaFold checkpoint (.pt). If provided, uses KernaFold + '
         'teacher for fully differentiable 16-channel reward (no Vienna API).')
argparser.add_argument(
    '--vienna_api_url', type=str,
    default='http://localhost:8000/jobs/analyze',
    help='URL of the Vienna RNA REST API endpoint')
argparser.add_argument(
    '--base_path', type=str, default=_DEFAULT_BASE_PATH,
    help='Base directory for saving results')

# Training hyperparameters
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=1000)
argparser.add_argument('--num_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps per optimizer update')
argparser.add_argument('--batch_size', type=int, default=16,
                        help='Samples per accumulation step (lower due to Vienna API overhead)')
argparser.add_argument('--save_every_n_epochs', type=int, default=50)

# DRAKES-specific hyperparameters
argparser.add_argument('--alpha', type=float, default=0.001,
                        help='KL divergence regularization weight')
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0,
                        help='Number of epochs for linear alpha warmup (0=no warmup)')
argparser.add_argument('--truncate_steps', type=int, default=50,
                        help='Number of diffusion steps with gradient flow')
argparser.add_argument('--total_num_steps', type=int, default=128,
                        help='Total diffusion sampling steps')
argparser.add_argument('--gumbel_temp', type=float, default=1.0,
                        help='Temperature for Gumbel-softmax sampling')
argparser.add_argument('--gradnorm_clip', type=float, default=1.0,
                        help='Max gradient norm for clipping')
argparser.add_argument('--copy_flag_temp', type=float, default=None,
                        help='Temperature for copy flag sigmoid (None=use raw prob)')
argparser.add_argument("--truncate_kl", type=str2bool, default=False,
                        help='Only compute KL for the last truncate_steps steps')
argparser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Multiply reward by this factor to amplify gradient signal')
argparser.add_argument('--kl_every_n', type=int, default=1,
                        help='Compute KL at every n-th timestep (saves memory, allows larger bs)')

# Evaluation
argparser.add_argument('--eval_every_n_epochs', type=int, default=50,
                        help='Run full evaluation every N epochs')
argparser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Number of sequences to generate for evaluation')

# Experiment
argparser.add_argument('--name', type=str, default='debug',
                        help='Experiment name (use "debug" to disable W&B)')
argparser.add_argument('--seed', type=int, default=0)

args = argparser.parse_args()
print(args)

# =============================================================================
# Setup: Hydra config, logging, model loading
# =============================================================================

# Reinitialize Hydra (may have been initialized by another script)
GlobalHydra.instance().clear()

# Compose the configuration from configs/config.yaml
initialize(config_path="configs", job_name="finetune_reward_bp")
cfg = compose(config_name="config.yaml")
cfg.eval.checkpoint_path = args.checkpoint_path

# Setup output directories and logging
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_base_dir = os.path.join(args.base_path, 'reward_bp_results')

if args.name == 'debug':
    print("Debug mode — W&B logging disabled")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = (
        f'alpha{args.alpha}_accum{args.num_accum_steps}_bsz{args.batch_size}'
        f'_truncate{args.truncate_steps}_temp{args.gumbel_temp}'
        f'_clip{args.gradnorm_clip}_{args.name}_{curr_time}'
    )
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(
        project='narry-kim-3utr-finetune',
        name=run_name,
        config=vars(args),
        dir=save_path,
    )
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# =============================================================================
# Load models
# =============================================================================

# Load pretrained diffusion model (two copies: one to fine-tune, one frozen for KL)
print(f'Loading pretrained diffusion model from {args.checkpoint_path}...')
new_model = diffusion_module.Diffusion.load_from_checkpoint(
    cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_module.Diffusion.load_from_checkpoint(
    cfg.eval.checkpoint_path, config=cfg)

# Freeze the old model (reference for KL divergence)
# Put on second GPU if available to free memory for larger batches
kl_device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
old_model.to(kl_device)
old_model.eval()
for param in old_model.parameters():
    param.requires_grad = False
print(f'  old_model (KL reference) on {kl_device}')

# Load the reward wrapper (KernaFold, distilled oracle, or Vienna-based)
if args.kernafold_checkpoint_path:
    from kernafold_reward_wrapper import KernaFoldRewardWrapper
    print(f'Loading KernaFoldRewardWrapper...')
    print(f'  KernaFold: {args.kernafold_checkpoint_path}')
    print(f'  Teacher: {args.regressor_checkpoint_dir}')
    reward_wrapper = KernaFoldRewardWrapper(
        kernafold_checkpoint_path=args.kernafold_checkpoint_path,
        regressor_checkpoint_dir=args.regressor_checkpoint_dir,
        device=str(new_model.device),
    )
elif args.distilled_oracle_path:
    from distilled_reward_wrapper import DistilledRewardWrapper
    print(f'Loading DistilledRewardWrapper from {args.distilled_oracle_path}...')
    reward_wrapper = DistilledRewardWrapper(
        checkpoint_path=args.distilled_oracle_path,
        device=str(new_model.device),
    )
    print(f'  Distilled oracle: d_input={reward_wrapper.config["d_input"]}, '
          f'pearson_vs_true={reward_wrapper.metrics.get("pearson_student_true", "?"):.4f}')
else:
    from vienna_reward_wrapper import ViennaRewardWrapper
    print(f'Loading ViennaRewardWrapper from {args.regressor_checkpoint_dir}...')
    reward_wrapper = ViennaRewardWrapper(
        regressor_checkpoint_dir=args.regressor_checkpoint_dir,
        device=str(new_model.device),
        vienna_api_url=args.vienna_api_url,
    )

# Load the evaluator (requires Vienna API for structural analysis)
print('Loading FinetuneEvaluator...')
evaluator = FinetuneEvaluator(vienna_api_url=args.vienna_api_url)

# Baseline evaluation (pretrained model before fine-tuning)
print('\n=== Baseline Evaluation (pretrained model) ===')
baseline_seqs = _generate_eval_sequences(
    new_model, args.eval_batch_size, args.total_num_steps,
    next(new_model.parameters()).device)
with torch.no_grad():
    baseline_oh = torch.zeros(
        len(baseline_seqs), 197, 4,
        device=next(new_model.parameters()).device)
    nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, seq in enumerate(baseline_seqs):
        for j, ch in enumerate(seq):
            baseline_oh[i, j, nt_to_idx[ch]] = 1.0
    baseline_rewards = reward_wrapper(baseline_oh).cpu().numpy()
baseline_metrics = evaluator.evaluate(baseline_seqs, baseline_rewards)
evaluator.print_report(baseline_metrics)
if args.name != 'debug':
    wandb.log({f'eval/{k}': v for k, v in baseline_metrics.items()
               if isinstance(v, (int, float))}, step=-1)

print('\nStarting DRAKES fine-tuning...')
fine_tune(new_model, reward_wrapper, old_model, args, log_path, save_path,
          evaluator=evaluator)
