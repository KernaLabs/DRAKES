"""Probe maximum batch size for fine-tuning at truncate_steps=50."""
import os, sys, torch, omegaconf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

omegaconf.OmegaConf.register_new_resolver('uuid', lambda: 'probe', use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

GlobalHydra.instance().clear()
initialize(config_path="configs", job_name="probe", version_base=None)
cfg = compose(config_name="config.yaml")
cfg.eval.checkpoint_path = 'experiments/checkpoints/best.ckpt'

import diffusion as diffusion_module
model = diffusion_module.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_module.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model.eval()
for p in old_model.parameters():
    p.requires_grad = False

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

model.config.finetuning.truncate_steps = 50
model.config.finetuning.gumbel_softmax_temp = 1.0
model.train()

for bs in [16, 24, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list = \
            model._sample_finetune_gradient(num_steps=128, eval_sp_size=bs, copy_flag_temp=None)
        reward = sample.sum(dim=(1, 2))
        total_kl = []
        for t in range(128):
            if t < 78: continue
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
        loss = -reward.mean() + 0.001 * kl_loss
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"bs={bs}, truncate=50: OK, peak={peak:.2f} GB", flush=True)
        model.zero_grad()
        del sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list, total_kl, kl_loss, loss, reward
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"bs={bs}, truncate=50: OOM, peak={peak:.2f} GB", flush=True)
        model.zero_grad()
        torch.cuda.empty_cache()
        break
