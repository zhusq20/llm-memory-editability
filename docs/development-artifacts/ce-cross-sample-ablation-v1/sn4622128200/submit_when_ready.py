"""Submit authorized continuation immediately after the image finishes building."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resume',HERE/'continue_four.py')
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
r.STATE.mkdir(mode=0o700,parents=True,exist_ok=True)
try:
    started=time.monotonic()
    while True:
        p=subprocess.run(['docker','--context','lm-memory','image','inspect',
            'lm-memory-ce-ablation:20261017','--format','{{.Id}}'],capture_output=True,text=True)
        if p.returncode==0:
            image=p.stdout.strip()
            break
        if time.monotonic()-started>3600:
            raise TimeoutError('Image build did not complete within one hour; no science submitted')
        r.save(r.STATE/'submission-state.json',dict(state='waiting_for_image_build',utc_epoch=time.time()))
        time.sleep(5)
    check=r.STATE/'runtime-checks'
    check.mkdir(mode=0o700,exist_ok=True)
    name='ce-ablation-sn8200-runtime-checks'
    command=['create','--name',name,'--user','1015:1016','--network','host','--cpus','4','--memory','8g',
        '--tmpfs','/tmp:rw,size=1g','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',f'type=bind,source={r.BASE},target=/persist',
        '--mount',f'type=bind,source={r.TOKEN},target=/run/secrets/wandb,readonly',
        '--workdir','/frozen',image,'python','/runtime_checks.py']
    # Runtime check writes to the persisted parent mount, avoiding root-squash on private subdirectories.
    code=(HERE/'runtime_checks.py').read_text().replace("'/out/",f"'/persist/{r.BATCH}/sn4622128200-continuation/runtime-checks/")
    path=r.STATE/'runtime_checks.py'
    path.write_text(code)
    r.docker(*command)
    r.docker('cp',str(path),f'{name}:/runtime_checks.py')
    with (r.ROOT/'cross-server-handoff/frozen-source.tar').open('rb') as f:
        r.docker('cp','-',f'{name}:/',input=f.read())
    r.docker('start',name)
    p=subprocess.run(['docker','--context','lm-memory','wait',name],capture_output=True,text=True,timeout=180)
    logs=r.docker('logs',name)
    (check/'runtime.log').write_bytes(logs.stdout+logs.stderr)
    assert p.returncode==0 and p.stdout.strip()=='0', 'Runtime/credential check failed; see preserved container'
    preflight=r.read(check/'result.json')
    assert preflight['passed'] is True
    r.save(r.STATE/'preflight.json',dict(passed=True,actual_image=image,
        original_image='sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee',
        source_sha256=r.MANIFEST,scientific_updates=0,
        repeated_development_or_engineering_updates=0,contract_suite_not_repeated=True,
        authorization='User requested immediate start without unrelated verification; same frozen source and pinned principal dependencies',
        runtime=preflight))
    r.save(r.STATE/'schedule-amendment.json',dict(host='sn4622128200',actual_image=image,
        original_configuration_unchanged=True,remaining_arms=r.ARMS,scientific_updates=10240,
        allocation='exclusive idle GPU among indices 1,2,3,4; max four concurrent, queue if occupied',
        admission='no compute processes, memory under 128MiB, utilization under 5%; zero-update CUDA check per arm',
        automatic_retry=False,utc_epoch=time.time()))
    r.save(r.STATE/'submission-state.json',dict(state='starting_controller',actual_image=image,utc_epoch=time.time()))
    candidates='GPU-364f0e00-7f5f-e6a7-9fea-03eb31c0bb3b,GPU-9c6ca0d7-11bb-e940-c845-efea3cc49c23,GPU-9dc24b81-029c-288b-ced5-ce32b996bae3,GPU-681ddfc6-46d1-0d48-ce5d-f9ec6cbf278c'
    os.execv(sys.executable,[sys.executable,str(HERE/'continue_four.py'),'--image',image,'--candidate-gpus',candidates])
except BaseException as exc:
    r.save(r.STATE/'submission-state.json',dict(state='needs_attention',error=repr(exc),traceback=traceback.format_exc()))
    raise
