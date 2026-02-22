"""Diagnose why DRAKES fine-tuning isn't working.

Check: how much gradient signal does the reward actually produce
through the sequence channels vs how much does KL produce?
"""
import torch, os, omegaconf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

omegaconf.OmegaConf.register_new_resolver('uuid', lambda: 'diag', use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
import drakes_paths as dp
omegaconf.OmegaConf.register_new_resolver('drakes_root', lambda: str(dp.storage_root), use_cache=True)

GlobalHydra.instance().clear()
initialize(config_path='configs', job_name='diag', version_base=None)
cfg = compose(config_name='config.yaml')
cfg.eval.checkpoint_path = str(dp.narry_kim.experiments_dir / 'checkpoints' / 'best.ckpt')

import diffusion as diffusion_module
from vienna_reward_wrapper import ViennaRewardWrapper

model = diffusion_module.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_module.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model.eval()
for p in old_model.parameters():
    p.requires_grad = False

wrapper = ViennaRewardWrapper(
    regressor_checkpoint_dir=str(dp.narry_kim.regressor_ckpt_dir),
    device=str(model.device),
)

model.config.finetuning.truncate_steps = 50
model.config.finetuning.gumbel_softmax_temp = 1.0
model.train()

print("=" * 60)
print("GRADIENT DIAGNOSIS")
print("=" * 60)

# Step 1: Sample with gradient
sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = \
    model._sample_finetune_gradient(num_steps=128, eval_sp_size=8, copy_flag_temp=None)

print(f"\nSample shape: {sample.shape}")
print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
print(f"Sample requires_grad: {sample.requires_grad}")

# Step 2: Compute reward gradient
reward = wrapper(sample)
reward_loss = -reward.mean()
print(f"\nReward values: {reward.detach().cpu().numpy()}")
print(f"Reward mean: {reward.mean().item():.4f}")

# Backprop reward only
reward_loss.backward(retain_graph=True)

# Check gradient on diffusion model params
reward_grad_norm = 0.0
reward_grad_norms_per_layer = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        gnorm = p.grad.norm().item()
        reward_grad_norm += gnorm ** 2
        reward_grad_norms_per_layer[name] = gnorm
reward_grad_norm = reward_grad_norm ** 0.5

print(f"\n--- REWARD GRADIENT ---")
print(f"Total grad norm from reward: {reward_grad_norm:.6f}")
print(f"Top 5 layers by grad norm:")
sorted_layers = sorted(reward_grad_norms_per_layer.items(), key=lambda x: -x[1])
for name, gnorm in sorted_layers[:5]:
    print(f"  {name}: {gnorm:.6f}")

# Save reward grads, then zero
reward_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
model.zero_grad()

# Step 3: Compute KL gradient
total_kl = []
for t in range(128):
    if t < 78:  # truncate_kl with truncate_steps=50
        continue
    last_x = last_x_list[t]
    condt = condt_list[t]
    move_chance_t = move_chance_t_list[t]
    copy_flag = copy_flag_list[t]
    log_p_x0 = model.forward(last_x, condt)[:, :, :-1]
    with torch.no_grad():
        log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]
    p_x0 = log_p_x0.exp()
    p_x0_old = log_p_x0_old.exp()
    kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0, 0, 0]
    kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
    total_kl.append(kl_div)

kl_loss = torch.stack(total_kl, 1).sum(1).mean()
print(f"\nKL loss value: {kl_loss.item():.4f}")

(0.001 * kl_loss).backward()

kl_grad_norm = 0.0
kl_grad_norms_per_layer = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        gnorm = p.grad.norm().item()
        kl_grad_norm += gnorm ** 2
        kl_grad_norms_per_layer[name] = gnorm
kl_grad_norm = kl_grad_norm ** 0.5

print(f"\n--- KL GRADIENT (alpha=0.001) ---")
print(f"Total grad norm from KL: {kl_grad_norm:.6f}")
print(f"Top 5 layers by grad norm:")
sorted_layers = sorted(kl_grad_norms_per_layer.items(), key=lambda x: -x[1])
for name, gnorm in sorted_layers[:5]:
    print(f"  {name}: {gnorm:.6f}")

# Compare
print(f"\n--- COMPARISON ---")
print(f"Reward grad norm: {reward_grad_norm:.6f}")
print(f"KL grad norm:     {kl_grad_norm:.6f}")
print(f"Ratio reward/KL:  {reward_grad_norm / max(kl_grad_norm, 1e-10):.4f}")

# Per-layer comparison
print(f"\nPer-layer reward/KL ratio:")
for name in sorted_layers[:10]:
    name = name[0]
    r = reward_grads.get(name)
    k = kl_grad_norms_per_layer.get(name, 0)
    rnorm = r.norm().item() if r is not None else 0
    print(f"  {name}: reward={rnorm:.6f} kl={k:.6f} ratio={rnorm/max(k,1e-10):.4f}")

# Step 4: Check gradient on soft_onehot directly
print(f"\n--- REGRESSOR INPUT SENSITIVITY ---")
# Create a fresh soft_onehot with grad
test_input = torch.randn(4, 197, 16, device=model.device, requires_grad=True)
# Fill with realistic values
with torch.no_grad():
    test_input[:, :, :4] = torch.randn(4, 197, 4).softmax(-1).to(model.device)
    test_input[:, :, 4:] = 0.0
test_input.requires_grad_(True)
test_reward = wrapper.regressor(test_input)
test_reward.mean().backward()

seq_grad = test_input.grad[:, :, :4].norm().item()
struct_grad = test_input.grad[:, :, 4:10].norm().item()
scalar_grad = test_input.grad[:, :, 10:16].norm().item()

print(f"Grad norm on seq channels (0:4):    {seq_grad:.6f}")
print(f"Grad norm on struct channels (4:10): {struct_grad:.6f}")
print(f"Grad norm on scalar channels (10:16): {scalar_grad:.6f}")
print(f"Ratio seq/(struct+scalar): {seq_grad / max(struct_grad + scalar_grad, 1e-10):.4f}")

# =============================================================================
# Step 5: KernaFold wrapper comparison
# =============================================================================
print(f"\n{'='*60}")
print("KERNAFOLD GRADIENT COMPARISON")
print(f"{'='*60}")

from kernafold_reward_wrapper import KernaFoldRewardWrapper
kf_wrapper = KernaFoldRewardWrapper(
    kernafold_checkpoint_path=str(dp.narry_kim.kernafold_ckpt),
    regressor_checkpoint_dir=str(dp.narry_kim.regressor_ckpt_dir),
    device=str(model.device),
)

# Re-sample with gradient for KernaFold (need fresh graph)
model.zero_grad()
sample_kf, kf_last_x_list, kf_condt_list, kf_move_chance_t_list, kf_copy_flag_list = \
    model._sample_finetune_gradient(num_steps=128, eval_sp_size=8, copy_flag_temp=None)

print(f"\nKF Sample shape: {sample_kf.shape}, requires_grad: {sample_kf.requires_grad}")

# Compute reward through KernaFold
kf_reward = kf_wrapper(sample_kf)
kf_reward_loss = -kf_reward.mean()
print(f"KF Reward values: {kf_reward.detach().cpu().numpy()}")
print(f"KF Reward mean: {kf_reward.mean().item():.4f}")

# Backprop reward only
kf_reward_loss.backward(retain_graph=True)

# Measure gradient norms on diffusion model params
kf_reward_grad_norm = 0.0
kf_reward_grad_norms_per_layer = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        gnorm = p.grad.norm().item()
        kf_reward_grad_norm += gnorm ** 2
        kf_reward_grad_norms_per_layer[name] = gnorm
kf_reward_grad_norm = kf_reward_grad_norm ** 0.5

print(f"\n--- KERNAFOLD REWARD GRADIENT ---")
print(f"Total grad norm from KF reward: {kf_reward_grad_norm:.6f}")
print(f"Top 5 layers by grad norm:")
kf_sorted = sorted(kf_reward_grad_norms_per_layer.items(), key=lambda x: -x[1])
for name, gnorm in kf_sorted[:5]:
    print(f"  {name}: {gnorm:.6f}")

# Compare Vienna vs KernaFold reward gradient
print(f"\n--- VIENNA vs KERNAFOLD COMPARISON ---")
print(f"Vienna reward grad norm:    {reward_grad_norm:.6f}")
print(f"KernaFold reward grad norm: {kf_reward_grad_norm:.6f}")
print(f"KF/Vienna ratio:            {kf_reward_grad_norm / max(reward_grad_norm, 1e-10):.4f}")

# Per-layer comparison
print(f"\nPer-layer Vienna vs KernaFold reward gradient:")
all_layers = sorted(set(list(reward_grads.keys()) + list(kf_reward_grad_norms_per_layer.keys())))
for name in all_layers[:15]:
    v = reward_grads.get(name)
    v_norm = v.norm().item() if v is not None else 0
    k_norm = kf_reward_grad_norms_per_layer.get(name, 0)
    print(f"  {name}: vienna={v_norm:.6f} kf={k_norm:.6f} ratio={k_norm/max(v_norm,1e-10):.2f}")

# Step 6: KernaFold per-channel gradient analysis
# Feed sample through KernaFold and measure gradient per channel group on the
# 16-channel regressor input (NOT the 4-channel diffusion output)
print(f"\n--- KERNAFOLD PER-CHANNEL GRADIENT ANALYSIS ---")
model.zero_grad()

# Create a sample that requires grad for channel analysis
sample_for_channels = sample_kf.detach().requires_grad_(True)
kf_reward_ch = kf_wrapper(sample_for_channels)
(-kf_reward_ch.mean()).backward()

# The gradient on sample_for_channels tells us how the reward changes w.r.t.
# each position of the 4-channel soft_onehot input
input_grad = sample_for_channels.grad
print(f"Gradient on 4-ch input: shape={input_grad.shape}")
print(f"  Total grad norm: {input_grad.norm().item():.6f}")
print(f"  Per-channel norms: A={input_grad[:,:,0].norm():.6f} "
      f"C={input_grad[:,:,1].norm():.6f} "
      f"G={input_grad[:,:,2].norm():.6f} "
      f"T={input_grad[:,:,3].norm():.6f}")
print(f"  Mean abs grad per position: {input_grad.abs().mean().item():.8f}")

# Also measure teacher input sensitivity with realistic KernaFold-produced features
print(f"\n--- TEACHER INPUT SENSITIVITY (with KernaFold features) ---")
# Build a realistic 16-channel input using KernaFold on the sample
with torch.no_grad():
    seq_st = sample_kf.detach()
    seq_hard = torch.nn.functional.one_hot(seq_st.argmax(-1), 4).float()
    if kf_wrapper.kernafold_variant == 'bpp':
        valid_pair_mask = kf_wrapper.kernafold._compute_valid_pair_mask(seq_hard)
        kf_out = kf_wrapper.kernafold(seq_hard, valid_pair_mask)
    else:
        kf_out = kf_wrapper.kernafold(seq_hard)
    mfe_oh = torch.nn.functional.one_hot(kf_out['mfe_logits'].argmax(-1), 3).float()
    cent_oh = torch.nn.functional.one_hot(kf_out['centroid_logits'].argmax(-1), 3).float()
    scalars_raw = kf_wrapper.kernafold.denormalize_scalars(kf_out['scalars_normed'])
    teacher_scalars = scalars_raw / kf_wrapper.scalar_divisors.to(scalars_raw.device)
    teacher_scalars = teacher_scalars.unsqueeze(1).expand(-1, 197, -1)
    realistic_input = torch.cat([seq_hard, mfe_oh, cent_oh, teacher_scalars], dim=-1)

# Now measure gradient through teacher on this realistic input
realistic_input = realistic_input.detach().requires_grad_(True)
teacher_reward = kf_wrapper.regressor(realistic_input)
teacher_reward.mean().backward()

rg = realistic_input.grad
seq_g = rg[:, :, :4].norm().item()
mfe_g = rg[:, :, 4:7].norm().item()
cent_g = rg[:, :, 7:10].norm().item()
scalar_g = rg[:, :, 10:16].norm().item()
total_g = rg.norm().item()

print(f"  Seq channels (0:4):     {seq_g:.6f} ({100*seq_g/total_g:.1f}%)")
print(f"  MFE struct (4:7):       {mfe_g:.6f} ({100*mfe_g/total_g:.1f}%)")
print(f"  Centroid struct (7:10): {cent_g:.6f} ({100*cent_g/total_g:.1f}%)")
print(f"  Scalars (10:16):        {scalar_g:.6f} ({100*scalar_g/total_g:.1f}%)")
print(f"  Total:                  {total_g:.6f}")
print(f"  Struct+Scalar / Seq:    {(mfe_g+cent_g+scalar_g)/max(seq_g,1e-10):.4f}")

# Compare with zero-features (what Vienna wrapper effectively gives for gradient)
print(f"\n--- COMPARISON: Realistic vs Zero features ---")
print(f"  With KF features:  seq={seq_g:.6f} struct={mfe_g+cent_g:.6f} scalar={scalar_g:.6f}")
# Re-run Vienna version for direct comparison
zero_input = realistic_input.detach().clone()
zero_input[:, :, 4:] = 0  # zero out struct+scalar like Vienna detached
zero_input = zero_input.requires_grad_(True)
teacher_reward_zero = kf_wrapper.regressor(zero_input)
teacher_reward_zero.mean().backward()
zg = zero_input.grad
print(f"  With zero features: seq={zg[:,:,:4].norm():.6f} struct={zg[:,:,4:10].norm():.6f} scalar={zg[:,:,10:16].norm():.6f}")

print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*60}")
