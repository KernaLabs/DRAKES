"""
Improved sampling methods for fine-tuning with better sequence quality.
"""

import torch
import torch.nn.functional as F


def straight_through_gumbel_softmax(logits, temperature=1.0, hard=True):
    """
    Gumbel-Softmax with straight-through estimator for better gradient flow.
    
    Args:
        logits: Unnormalized log probabilities [batch, seq_len, vocab]
        temperature: Temperature for Gumbel-Softmax (lower = more discrete)
        hard: If True, use straight-through estimator for discrete samples
    
    Returns:
        Samples that are discrete in forward pass but continuous in backward
    """
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    
    # Apply temperature and softmax
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # Straight-through: discrete forward, continuous backward
        y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=logits.shape[-1]).float()
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    
    return y


def improved_sample_finetune_gradient(self, num_steps=None, eps=1e-5, eval_sp_size=None, 
                                     use_hard_samples=False, temperature_schedule='constant'):
    """
    Improved sampling for fine-tuning with better sequence quality.
    
    Args:
        num_steps: Number of diffusion steps
        eps: Small constant for numerical stability
        eval_sp_size: Evaluation sample size (batch size)
        use_hard_samples: If True, use hard (discrete) samples for evaluation
        temperature_schedule: 'constant', 'linear_decay', or 'exponential_decay'
    
    Returns:
        samples: Generated samples (can be soft or hard)
        last_x_list: List of states at each timestep
        condt_list: List of conditioning at each timestep
        move_chance_t_list: List of move chances
        copy_flag_list: List of copy flags
    """
    assert self.parameterization == 'subs' and self.sampler == 'ddpm'
    
    if eval_sp_size is None:
        batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
        batch_size_per_gpu = eval_sp_size
    
    if num_steps is None:
        num_steps = self.config.sampling.steps
    
    # Initialize with prior
    x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    
    last_x_list = []
    condt_list = []
    move_chance_t_list = []
    copy_flag_list = []
    
    # Get base temperature from config
    base_temp = self.config.finetuning.gumbel_softmax_temp
    
    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
        
        # Calculate temperature for this step
        if temperature_schedule == 'linear_decay':
            current_temp = base_temp * (1.0 - i / num_steps) + 0.1  # Decay to 0.1
        elif temperature_schedule == 'exponential_decay':
            current_temp = base_temp * (0.1 ** (i / num_steps))  # Exponential decay
        else:  # constant
            current_temp = base_temp
        
        if i < num_steps - self.config.finetuning.truncate_steps:
            # Regular DDPM update (no gradients needed)
            with torch.no_grad():
                x, last_x, condt, move_chance_t, copy_flag = self._ddpm_update(x, t, dt, return_process=True)
                x = x.detach()
                copy_flag = copy_flag.unsqueeze(-1)
                last_x = F.one_hot(last_x, num_classes=self.vocab_size).to(torch.float32).detach()
        else:
            # Fine-tuning gradient update with improved sampling
            x, last_x, condt, move_chance_t, copy_flag = improved_ddpm_update_finetune(
                self, x, t, dt, current_temp, use_hard_samples, return_process=True
            )
        
        last_x_list.append(last_x)
        condt_list.append(condt)
        move_chance_t_list.append(move_chance_t)
        copy_flag_list.append(copy_flag)
    
    return x, last_x_list, condt_list, move_chance_t_list, copy_flag_list


def improved_ddpm_update_finetune(self, x, t, dt, temperature, use_hard_samples, return_process=False):
    """
    Improved DDPM update for fine-tuning with better sampling quality.
    """
    # Ensure x is in the right format
    if x.ndim == 2 or x.shape[-1] != self.vocab_size:
        x_one_hot = F.one_hot(x, num_classes=self.vocab_size).to(torch.float32)
        x_indices = x
    else:
        x_one_hot = x
        x_indices = x.argmax(dim=-1)
    
    # Get noise parameters
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
    
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    
    # Forward pass
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x_indices, unet_conditioning)
    
    # Calculate transition probabilities
    q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
    
    # Set mask token probability
    if q_xs.shape[-1] > self.mask_index:
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    
    # Improved sampling with straight-through estimator
    if use_hard_samples:
        # Use straight-through Gumbel-Softmax for better gradients
        _x = straight_through_gumbel_softmax(
            torch.log(q_xs + 1e-20), 
            temperature=temperature, 
            hard=True
        )
    else:
        # Original soft sampling
        _x = _sample_categorical_gradient(q_xs, temp=temperature)
    
    # Calculate copy flags with improved smoothness
    if x_one_hot.shape[-1] > self.mask_index:
        is_masked = x_one_hot[:, :, self.mask_index]
        # Smoother transition for copy flags
        copy_flag_logit = 10.0 * (1.0 - is_masked)  # Stronger signal
        soft_copy_flag = torch.sigmoid(copy_flag_logit).unsqueeze(-1)
    else:
        soft_copy_flag = torch.ones_like(x_one_hot[:, :, 0]).unsqueeze(-1)
    
    # Mix old and new based on copy flag
    x_new = soft_copy_flag * x_one_hot + (1 - soft_copy_flag) * _x
    
    if return_process:
        return x_new, x_one_hot, unet_conditioning, move_chance_t, soft_copy_flag
    else:
        return x_new


def _sample_categorical_gradient(categorical_probs, temp=1.0):
    """Original soft categorical sampling for comparison."""
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    output = torch.nn.functional.softmax(
        (torch.log(categorical_probs + 1e-20) - torch.log(gumbel_norm)) / temp, 
        dim=-1
    )
    return output


# Monkey-patch the diffusion module with improved methods
def patch_diffusion_module(diffusion_module):
    """
    Patch the diffusion module with improved sampling methods.
    
    Usage:
        import diffusion
        from diffusion_improved_sampling import patch_diffusion_module
        patch_diffusion_module(diffusion)
    """
    # Add the improved methods to the Diffusion class
    diffusion_module.Diffusion.improved_sample_finetune_gradient = improved_sample_finetune_gradient
    diffusion_module.Diffusion.improved_ddpm_update_finetune = improved_ddpm_update_finetune
    
    # Also add the helper functions to the module
    diffusion_module.straight_through_gumbel_softmax = straight_through_gumbel_softmax
    
    print("Patched diffusion module with improved sampling methods")