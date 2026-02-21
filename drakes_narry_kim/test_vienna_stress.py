"""Stress-test Vienna API with generated sequences to reproduce 500 errors."""
import torch, os, requests, omegaconf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

omegaconf.OmegaConf.register_new_resolver('uuid', lambda: 'test', use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

GlobalHydra.instance().clear()
initialize(config_path='configs', job_name='test', version_base=None)
cfg = compose(config_name='config.yaml')
cfg.eval.checkpoint_path = 'experiments/checkpoints/best.ckpt'

import diffusion as diffusion_module
model = diffusion_module.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
model.eval()

idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
api_url = 'http://localhost:8000/jobs/analyze'

n_fail = 0
n_ok = 0
for batch_i in range(7):
    with torch.no_grad():
        x = model.mask_index * torch.ones(32, 197, dtype=torch.int64, device=model.device)
        timesteps = torch.linspace(1, 1e-3, 129, device=model.device)
        dt = (1 - 1e-3) / 128
        for i in range(128):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
            x = model._ddpm_update(x, t, dt)

    for j in range(32):
        seq = ''.join(idx_to_nt.get(x[j, k].item(), 'A') for k in range(197))
        try:
            resp = requests.post(api_url, json={'sequence': seq, 'use_n1m': True}, timeout=30)
            resp.raise_for_status()
            n_ok += 1
        except Exception as e:
            n_fail += 1
            status = getattr(resp, 'status_code', '?')
            body = ''
            try:
                body = resp.text[:200]
            except:
                pass
            print(f'FAIL #{n_fail}: {seq}')
            print(f'  status={status} err={e}')
            print(f'  body={body}')
    print(f'Batch {batch_i}: {n_ok} ok, {n_fail} fail so far', flush=True)

print(f'\nTOTAL: {n_ok} ok, {n_fail} fail out of {n_ok+n_fail}')
