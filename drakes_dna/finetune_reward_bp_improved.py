# Direct reward backpropagation with improved sampling
import diffusion
import dataloader
from diffusion_improved_sampling import patch_diffusion_module, straight_through_gumbel_softmax
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
from utils import str2bool, set_seed
import drakes_paths as dp


class GCContentRewardModel(torch.nn.Module):
    """Reward model based on GC content of DNA sequences."""
    
    def __init__(self, device='cuda', target_gc=0.5):
        super().__init__()
        self.device = device
        self.target_gc = target_gc
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape [batch_size, 4, seq_len] (one-hot encoded DNA)
        Returns:
            rewards: tensor of shape [batch_size, 3] with GC content-based rewards
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        # Calculate GC content (C is index 1, G is index 2)
        gc_count = x[:, 6, :].sum(dim=1) + x[:, 7, :].sum(dim=1)
        gc_content = gc_count / seq_len
        
        # Reward based on distance from target GC content
        # Higher reward for sequences closer to target
        gc_distance = torch.abs(gc_content - self.target_gc)
        rewards = 1.0 - gc_distance  # Max reward of 1.0 at target GC
        
        # Return same reward for all 3 tasks
        rewards = rewards.unsqueeze(1).expand(batch_size, 3)
        return rewards.unsqueeze(-1)
    
    def eval(self):
        return self
    
    def to(self, device):
        self.device = device
        return self


def fine_tune_improved(new_model, reward_model, reward_model_eval, old_model, args, eps=1e-5):
    """Fine-tuning with improved sampling for better sequence quality."""
    
    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')
    
    # Add finetuning config if needed
    from omegaconf import OmegaConf
    if not hasattr(new_model.config, 'finetuning'):
        OmegaConf.set_struct(new_model.config, False)
        new_model.config.finetuning = OmegaConf.create({
            'truncate_steps': args.truncate_steps,
            'gumbel_softmax_temp': args.gumbel_temp
        })
        OmegaConf.set_struct(new_model.config, True)
    else:
        new_model.config.finetuning.truncate_steps = args.truncate_steps
        new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    
    dt = (1 - eps) / args.total_num_steps
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    batch_losses = []
    batch_rewards = []
    
    for epoch_num in range(args.num_epochs):
        rewards = []
        rewards_eval = []
        rewards_hard = []  # Track rewards from hard samples
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_model.train()
        
        for _step in range(args.num_accum_steps):
            # Choose sampling strategy based on args
            if hasattr(args, 'use_improved_sampling') and args.use_improved_sampling:
                # Use improved sampling with temperature scheduling
                temp_schedule = args.temperature_schedule if hasattr(args, 'temperature_schedule') else 'constant'
                use_hard = args.use_hard_samples if hasattr(args, 'use_hard_samples') else False
                
                sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = \
                    new_model.improved_sample_finetune_gradient(
                        eval_sp_size=args.batch_size,
                        use_hard_samples=use_hard,
                        temperature_schedule=temp_schedule
                    )
            else:
                # Original sampling
                sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = \
                    new_model._sample_finetune_gradient(
                        eval_sp_size=args.batch_size, 
                        copy_flag_temp=args.copy_flag_temp
                    )
            
            # Extract DNA bases properly
            if sample.shape[-1] > 4:
                # Sample contains full vocabulary, extract DNA bases
                dna_indices = [6, 7, 8, 9]  # A, C, G, T
                sample_dna = sample[:, :, dna_indices]
                sample_dna = sample_dna / sample_dna.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                sample_dna = sample
            
            # For training: use soft samples
            sample2 = torch.transpose(sample_dna, 1, 2)
            preds = reward_model(sample2).squeeze(-1)
            reward = preds[:, 0]
            
            # For evaluation: always use hard samples
            if args.use_straight_through:
                # Use straight-through estimator for better gradients
                sample_hard = straight_through_gumbel_softmax(
                    torch.log(sample_dna + 1e-20),
                    temperature=args.eval_temp,
                    hard=True
                )
                sample_hard_t = torch.transpose(sample_hard, 1, 2)
            else:
                # Standard argmax
                sample_argmax = torch.argmax(sample_dna, 2)
                sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes=4)
                sample_hard_t = torch.transpose(sample_argmax, 1, 2)
            
            # Evaluate with hard samples
            preds_hard = reward_model(sample_hard_t).squeeze(-1)
            reward_hard = preds_hard[:, 0]
            rewards_hard.append(reward_hard.detach().cpu().numpy())
            
            preds_eval = reward_model_eval(sample_hard_t).squeeze(-1)
            reward_eval = preds_eval[:, 0]
            rewards_eval.append(reward_eval.detach().cpu().numpy())
            
            # KL divergence calculation
            total_kl = []
            for random_t in range(args.total_num_steps):
                if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
                    continue
                
                last_x = last_x_list[random_t]
                condt = condt_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                copy_flag = copy_flag_list[random_t]
                
                # Handle different input formats
                if last_x.ndim == 3 and last_x.shape[-1] > 1:
                    last_x_indices = last_x.argmax(dim=-1)
                else:
                    last_x_indices = last_x
                
                log_p_x0 = new_model.forward(last_x_indices, condt)
                log_p_x0_old = old_model.forward(last_x_indices, condt)
                
                # Remove mask token if present
                if log_p_x0.shape[-1] > new_model.vocab_size - 1:
                    log_p_x0 = log_p_x0[:, :, :-1]
                    log_p_x0_old = log_p_x0_old[:, :, :-1]
                
                p_x0 = log_p_x0.exp()
                p_x0_old = log_p_x0_old.exp()
                
                kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0,0,0]
                
                # Proper masking
                if last_x_indices.ndim == 2:
                    last_x_oh = F.one_hot(last_x_indices, num_classes=new_model.vocab_size).to(torch.float32)
                    if last_x_oh.shape[-1] > p_x0.shape[-1]:
                        last_x_oh = last_x_oh[:, :, :-1]
                else:
                    last_x_oh = last_x
                    if last_x_oh.shape[-1] > p_x0.shape[-1]:
                        last_x_oh = last_x_oh[:, :, :-1]
                
                kl_div = (kl_div * last_x_oh).sum((1, 2))
                total_kl.append(kl_div)
            
            # Calculate losses with warmup
            if epoch_num < args.alpha_schedule_warmup:
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha
            
            kl_loss = torch.stack(total_kl, 1).sum(1).mean()
            reward_loss = -torch.mean(reward)
            loss = reward_loss + kl_loss * current_alpha
            loss = loss / args.num_accum_steps
            
            loss.backward()
            
            if (_step + 1) % args.num_accum_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()
            
            batch_losses.append(loss.cpu().detach().numpy())
            batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            rewards.append(reward.detach().cpu().numpy())
            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            reward_losses.append(reward_loss.cpu().detach().numpy())
            kl_losses.append(kl_loss.cpu().detach().numpy())
        
        # Convert to arrays
        rewards = np.array(rewards).flatten()
        rewards_eval = np.array(rewards_eval).flatten()
        rewards_hard = np.array(rewards_hard).flatten()
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        kl_losses = np.array(kl_losses)
        
        print(f"Epoch {epoch_num}: "
              f"Reward (soft)={np.mean(rewards):.4f}, "
              f"Reward (hard)={np.mean(rewards_hard):.4f}, "
              f"Reward (eval)={np.mean(rewards_eval):.4f}, "
              f"Loss={np.mean(losses):.4f}, "
              f"KL={np.mean(kl_losses):.4f}")
        
        if args.name != 'debug':
            wandb.log({
                "epoch": epoch_num, 
                "mean_reward_soft": np.mean(rewards),
                "mean_reward_hard": np.mean(rewards_hard),
                "mean_reward_eval": np.mean(rewards_eval),
                "mean_grad_norm": tot_grad_norm,
                "mean_loss": np.mean(losses),
                "mean_reward_loss": np.mean(reward_losses),
                "mean_kl_loss": np.mean(kl_losses)
            })
        
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch_num} "
                   f"Reward_soft={np.mean(rewards):.4f} "
                   f"Reward_hard={np.mean(rewards_hard):.4f} "
                   f"Reward_eval={np.mean(rewards_eval):.4f} "
                   f"Loss={np.mean(losses):.4f} "
                   f"KL={np.mean(kl_losses):.4f}\n")
        
        if (epoch_num + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
    
    if args.name != 'debug':
        wandb.finish()
    
    return batch_losses


# Argument parser
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default=None, help='(deprecated, uses drakes_paths)')
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=10)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=8)
argparser.add_argument('--name', type=str, default='improved_sampling_test')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=5)
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument('--checkpoint_path', type=str, default=None)

# New arguments for improved sampling
argparser.add_argument('--use_improved_sampling', type=str2bool, default=True,
                      help='Use improved sampling methods')
argparser.add_argument('--use_hard_samples', type=str2bool, default=False,
                      help='Use hard samples during training (straight-through)')
argparser.add_argument('--use_straight_through', type=str2bool, default=True,
                      help='Use straight-through estimator for evaluation')
argparser.add_argument('--temperature_schedule', type=str, default='linear_decay',
                      choices=['constant', 'linear_decay', 'exponential_decay'],
                      help='Temperature scheduling strategy')
argparser.add_argument('--eval_temp', type=float, default=0.5,
                      help='Temperature for evaluation sampling')
argparser.add_argument('--target_gc', type=float, default=0.5,
                      help='Target GC content for reward model')

args = argparser.parse_args()
print(args)

# Checkpoint path
if args.checkpoint_path is not None:
    CKPT_PATH = args.checkpoint_path
else:
    CKPT_PATH = str(dp.dna.pretrained_ckpt)

print(f"Using checkpoint: {CKPT_PATH}")

log_base_dir = str(dp.dna.reward_bp_results_improved)

# Initialize Hydra
GlobalHydra.instance().clear()
initialize(config_path="configs", job_name="load_model")
cfg = compose(config_name="ginkgo_3utr.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup logging
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = f'improved_{args.temperature_schedule}_temp{args.gumbel_temp}_{args.name}_{curr_time}'
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='reward_bp_improved', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

print("Loading models...")

# Check checkpoint
if not os.path.exists(CKPT_PATH):
    print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
    exit(1)

try:
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(cfg)
    
    # Patch diffusion module with improved methods
    patch_diffusion_module(diffusion)
    
    # Load models
    new_model = diffusion.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, tokenizer=tokenizer, config=cfg)
    old_model = diffusion.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, tokenizer=tokenizer, config=cfg)
    
    # Use GC content-based reward models
    print(f"Using GC content reward model with target={args.target_gc}")
    reward_model = GCContentRewardModel(device=new_model.device, target_gc=args.target_gc)
    reward_model_eval = GCContentRewardModel(device=new_model.device, target_gc=args.target_gc)
    
    print("Models loaded successfully!")
    print(f"New model device: {new_model.device}")
    print(f"Reward model device: {reward_model.device}")
    
    # Run fine-tuning with improved sampling
    fine_tune_improved(new_model, reward_model, reward_model_eval, old_model, args)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    raise