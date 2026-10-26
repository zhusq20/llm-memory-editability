"""Zero-science-update checks inside the new immutable image."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['WANDB_CONFIG_DIR']='/tmp/wandb-config'
os.environ['WANDB_CACHE_DIR']='/tmp/wandb-cache'
import torch
import wandb

EXPECTED = {'torch':'2.8.0','transformers':'4.57.3','numpy':'2.2.6','pytest':'9.1.1',
            'ruff':'0.14.0','wandb':'0.30.0','huggingface-hub':'0.36.0'}
versions = {k:importlib.metadata.version(k) for k in EXPECTED}
assert all(versions[k].split('+')[0] == v for k,v in EXPECTED.items())
assert torch.version.cuda == '12.8'
os.environ['WANDB_API_KEY'] = Path('/run/secrets/wandb').read_text().strip()
api = wandb.Api(timeout=60)
viewer = api.viewer
assert viewer, 'W&B credentials could not be authenticated'
project = api.project('llm-memory-editability', entity='zhusq20')
assert project is not None
result = {'versions':versions,'cuda':torch.version.cuda,'wandb_authenticated':True,
          'project':'zhusq20/llm-memory-editability','scientific_optimizer_updates':0}
Path('/out/runtime.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run([sys.executable,'-m','pip','freeze'],stdout=Path('/out/requirements.txt').open('w'),check=True)
result['passed']=True
Path('/out/result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
