# direct reward backpropagation with random reward function
import diffusion
import dataloader
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


class RandomRewardModel(torch.nn.Module):
    """A dummy reward model that returns random rewards for testing."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape [batch_size, 4, seq_len] (one-hot encoded DNA)
               where the 4 channels correspond to A, C, G, T (indices 0, 1, 2, 3)
        Returns:
            rewards: tensor of shape [batch_size, 3] with random rewards
        """
        batch_size = x.shape[0]
        
        # Generate random rewards for testing
        # This is a dummy reward model that returns random values
        # In a real scenario, this would be replaced with an actual reward function
        rewards = torch.rand(batch_size, 3, device=self.device)
        
        return rewards.unsqueeze(-1)  # Add extra dimension to match expected shape
    
    def eval(self):
        return self
    
    def to(self, device):
        self.device = device
        return self


def fine_tune(new_model, new_model_y, new_model_y_eval, old_model, args, eps=1e-5):

    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    # Add finetuning config if it doesn't exist
    from omegaconf import OmegaConf
    if not hasattr(new_model.config, 'finetuning'):
        # Temporarily disable struct mode to allow adding new keys
        OmegaConf.set_struct(new_model.config, False)
        new_model.config.finetuning = OmegaConf.create({
            'truncate_steps': args.truncate_steps,
            'gumbel_softmax_temp': args.gumbel_temp
        })
        # Re-enable struct mode
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
        losses = []
        reward_losses = []
        kl_losses = []
        tot_grad_norm = 0.0
        new_model.train()
        for _step in range(args.num_accum_steps):
            sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = new_model._sample_finetune_gradient(eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp) # [bsz, seqlen, vocab_size]
            
            # Extract only DNA bases (indices 6-9: A, C, G, T)
            # The sample now contains probabilities for all vocabulary tokens
            # We need to extract and renormalize just the DNA bases
            dna_indices = [6, 7, 8, 9]  # A, C, G, T in the tokenizer
            sample_dna = sample[:, :, dna_indices]  # [bsz, seqlen, 4]
            
            # Renormalize to ensure valid probability distribution
            sample_dna = sample_dna / sample_dna.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
            sample2 = torch.transpose(sample_dna, 1, 2)  # [bsz, 4, seqlen]
            preds = new_model_y(sample2).squeeze(-1) # [bsz, 3]
            reward = preds[:, 0]

            sample_argmax = torch.argmax(sample_dna, 2)
            sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes=4)
            sample_argmax = torch.transpose(sample_argmax, 1, 2)

            preds_argmax = new_model_y(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
            rewards.append(reward_argmax.detach().cpu().numpy())
            
            preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
            reward_argmax_eval = preds_eval[:, 0]
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            
            total_kl = []
            
            # calculate the KL divergence
            for random_t in range(args.total_num_steps):
                if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
                    continue
                last_x = last_x_list[random_t] # [bsz, seqlen, vocab_size]
                condt = condt_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                copy_flag = copy_flag_list[random_t] # [bsz, seqlen, 1]
                
                # Forward pass expects indices, not one-hot
                if last_x.ndim == 3 and last_x.shape[-1] > 1:
                    # last_x is one-hot, convert to indices
                    last_x_indices = last_x.argmax(dim=-1)
                else:
                    last_x_indices = last_x
                    
                log_p_x0 = new_model.forward(last_x_indices, condt)
                log_p_x0_old = old_model.forward(last_x_indices, condt)
                
                # Remove mask token probability if present
                if log_p_x0.shape[-1] > new_model.vocab_size - 1:
                    log_p_x0 = log_p_x0[:, :, :-1]
                    log_p_x0_old = log_p_x0_old[:, :, :-1]

                p_x0 = log_p_x0.exp() # [bsz, seqlen, 4]
                p_x0_old = log_p_x0_old.exp()

                kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0,0,0]
                
                # Convert last_x to one-hot if needed for masking
                if last_x_indices.ndim == 2:
                    last_x_oh = F.one_hot(last_x_indices, num_classes=new_model.vocab_size).to(torch.float32)
                    if last_x_oh.shape[-1] > p_x0.shape[-1]:
                        last_x_oh = last_x_oh[:, :, :-1]  # Remove mask dimension
                else:
                    last_x_oh = last_x
                    if last_x_oh.shape[-1] > p_x0.shape[-1]:
                        last_x_oh = last_x_oh[:, :, :-1]
                        
                kl_div = (kl_div * last_x_oh).sum((1, 2)) # [bsz]
                total_kl.append(kl_div)

            if epoch_num < args.alpha_schedule_warmup:
                # linear warmup
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha

            kl_loss = torch.stack(total_kl, 1).sum(1).mean()
            reward_loss = - torch.mean(reward)
            loss = reward_loss + kl_loss * current_alpha
            loss = loss / args.num_accum_steps
            
            loss.backward()
            if (_step + 1) % args.num_accum_steps == 0: # Gradient accumulation
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            batch_losses.append(loss.cpu().detach().numpy())
            batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            reward_losses.append(reward_loss.cpu().detach().numpy())
            kl_losses.append(kl_loss.cpu().detach().numpy())
        
        rewards = np.array(rewards)
        rewards_eval = np.array(rewards_eval)
        losses = np.array(losses)
        reward_losses = np.array(reward_losses)
        kl_losses = np.array(kl_losses)

        print("Epoch %d"%epoch_num, "Mean reward %f"%np.mean(rewards), "Mean reward eval %f"%np.mean(rewards_eval), 
        "Mean grad norm %f"%tot_grad_norm, "Mean loss %f"%np.mean(losses), "Mean reward loss %f"%np.mean(reward_losses), "Mean kl loss %f"%np.mean(kl_losses))
        if args.name != 'debug':
            wandb.log({"epoch": epoch_num, "mean_reward": np.mean(rewards), "mean_reward_eval": np.mean(rewards_eval), 
            "mean_grad_norm": tot_grad_norm, "mean_loss": np.mean(losses), "mean reward loss": np.mean(reward_losses), "mean kl loss": np.mean(kl_losses)})
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch_num} Mean reward {np.mean(rewards)} Mean reward eval {np.mean(rewards_eval)} Mean grad norm {tot_grad_norm} Mean loss {np.mean(losses)} Mean reward loss {np.mean(reward_losses)} Mean kl loss {np.mean(kl_losses)}\n")
        
        if (epoch_num+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
    
    if args.name != 'debug':
        wandb.finish()

    return batch_losses

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='/mnt/ssd1/code/DRAKES/drakes_dna/')
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=10)  # Reduced for testing
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=8)  # Reduced for testing
argparser.add_argument('--name', type=str, default='random_reward_test')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=5)  # Save more frequently for testing
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
# Add option to specify checkpoint path
argparser.add_argument('--checkpoint_path', type=str, default=None, 
                       help='Path to pretrained checkpoint. If not specified, uses default path.')
args = argparser.parse_args()
print(args)

# pretrained model path - now configurable
if args.checkpoint_path is not None:
    CKPT_PATH = args.checkpoint_path
else:
    CKPT_PATH = '/mnt/ssd1/code/mdlm/ginkgo_3utr_experiments/checkpoints/last.ckpt'

print(f"Using checkpoint: {CKPT_PATH}")

log_base_dir = os.path.join(args.base_path, 'reward_bp_results_random')

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs", job_name="load_model")
cfg = compose(config_name="ginkgo_3utr.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# initialize a log file
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    run_name = f'alpha{args.alpha}_accum{args.num_accum_steps}_bsz{args.batch_size}_truncate{args.truncate_steps}_temp{args.gumbel_temp}_clip{args.gradnorm_clip}_{args.name}_{curr_time}'
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='reward_bp_random_test', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

print("Loading models...")

# Check if checkpoint exists
if not os.path.exists(CKPT_PATH):
    print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
    print("Available checkpoints in ginkgo_3utr_experiments:")
    ckpt_dir = "/mnt/ssd1/code/mdlm/ginkgo_3utr_experiments/checkpoints"
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith('.ckpt'):
                print(f"  - {os.path.join(ckpt_dir, f)}")
    exit(1)

try:
    # Initialize the models
    # Get tokenizer for DNA sequences 
    tokenizer = dataloader.get_tokenizer(cfg)
    
    new_model = diffusion.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, tokenizer=tokenizer, config=cfg)
    old_model = diffusion.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, tokenizer=tokenizer, config=cfg)
    
    # Use GC content-based reward models instead of real oracle
    print("Using GC content-based reward models for testing...")
    reward_model = RandomRewardModel(device=new_model.device)
    reward_model_eval = RandomRewardModel(device=new_model.device)
    
    print("Models loaded successfully!")
    print(f"New model device: {new_model.device}")
    print(f"Reward model device: {reward_model.device}")
    
    fine_tune(new_model, reward_model, reward_model_eval, old_model, args)
    
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"Tried to load checkpoint from: {cfg.eval.checkpoint_path}")
    raise