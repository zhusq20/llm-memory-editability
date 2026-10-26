"""Resume only four authorized arms using unchanged source; never retry a stage."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

BATCH = 'ce-cross-sample-ablation-v1'
BASE = Path('/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability')
ROOT = BASE / BATCH
STATE = ROOT / 'sn4622128200-continuation'
TOKEN = Path('/home/siqizhu4/.cache/lm-memory-ce-ablation/wandb-token')
MANIFEST = '79a5cf4cf4870527c2d6653e39d86a647ea991ae8bcf4204208f4eb36154ad83'
ARMS = ('loop-carry-adam', 'discoloop-lower-lr', 'discoloop-carry-adam', 'discoloop-no-bound')
MUTEX = threading.Lock()
LEASED = set()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def docker(*args, **kwargs):
    return subprocess.run(['docker', '--context', 'lm-memory', *args], check=True,
                          capture_output=True, timeout=120, **kwargs)


def inventory():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True,
                       text=True, timeout=20)
    result = []
    for line in p.stdout.splitlines():
        idx, uid, used, util = [v.strip() for v in line.split(',')]
        result.append(dict(index=int(idx), uuid=uid, memory_used=int(used), utilization=int(util)))
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                        '--format=csv,noheader'], check=True, capture_output=True,
                       text=True, timeout=20)
    occupied = set(p.stdout.splitlines())
    return [r for r in result if r['uuid'] not in occupied and r['memory_used'] < 128
            and r['utilization'] < 5]


def lease(arm, allowed):
    started = time.monotonic()
    while time.monotonic() - started < 24 * 3600:
        with MUTEX:
            for gpu in inventory():
                if gpu['uuid'] in allowed and gpu['uuid'] not in LEASED:
                    lock = (STATE / ('gpu-' + gpu['uuid'] + '.lock')).open('a')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    LEASED.add(gpu['uuid'])
                    return gpu, lock
        save(STATE / arm / 'status.json', dict(state='waiting_for_idle_verified_gpu', utc_epoch=time.time()))
        time.sleep(15)
    raise TimeoutError('No verified idle GPU became available; other work preserved')


def existing(arm, stage):
    d = ROOT / arm
    markers = [d / f'launch-{stage}.json', d / f'exit-{stage}.json', d / 'output' / stage / 'result.json']
    if not any(p.exists() for p in markers):
        if (d / 'output' / stage).exists() and any((d / 'output' / stage).iterdir()):
            raise RuntimeError(f'Unfinished stage evidence: {arm}/{stage}')
        return False
    assert all(p.is_file() for p in markers), f'Partial stage preserved: {arm}/{stage}'
    launch, exit_, result = map(read, markers)
    assert launch['source_sha256'] == MANIFEST and exit_['ExitCode'] == 0 and result['passed'] is True
    return True


def stage(arm, name, image, gpu=None, command=None, timeout=43200):
    if name != 'report' and existing(arm, name):
        return
    d = ROOT / arm
    if name == 'report':
        assert not (d / 'launch-report.json').exists()
        assert not (d / 'output/result.json').exists()
    container = f'ce-ablation-sn8200-v2-{arm}-{name}'
    cpu = {'loop-carry-adam':'64-67','discoloop-lower-lr':'68-71',
           'discoloop-carry-adam':'72-75','discoloop-no-bound':'76-79'}.get(arm, '64-67')
    args = ['create', '--name', container, '--label', f'lm.batch={BATCH}',
            '--label', f'lm.arm={arm}', '--label', f'lm.stage={name}',
            '--user', '1015:1016', '--network', 'host', '--cpuset-cpus', cpu,
            '--cpus', '4', '--memory', '48g', '--tmpfs', '/tmp:rw,size=2g',
            '--mount', f'type=bind,source={BASE.parent},target=/ossfs/workspace,readonly',
            '--mount', f'type=bind,source={BASE},target=/persist',
            '--mount', f'type=bind,source={TOKEN},target=/run/secrets/wandb,readonly',
            '--workdir', '/frozen']
    for item in ['PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/frozen/src:/frozen/scripts',
                 f'LM_SOURCE_MANIFEST_SHA256={MANIFEST}', f'LM_RUNTIME_IMAGE={image}',
                 'XDG_CACHE_HOME=/tmp/cache','WANDB_CACHE_DIR=/tmp/wandb-cache',
                 'WANDB_CONFIG_DIR=/tmp/wandb-config','WANDB_DATA_DIR=/tmp/wandb-data']:
        args += ['--env', item]
    if gpu:
        admission_start = time.monotonic()
        while gpu['uuid'] not in {r['uuid'] for r in inventory()}:
            if time.monotonic() - admission_start > 120:
                raise RuntimeError('GPU became occupied before launch; external work preserved')
            time.sleep(5)
        args += ['--gpus', 'device=' + gpu['uuid']]
    if command is None:
        command = ['python','scripts/run_ce_cross_sample_ablation.py',name,'--arm',arm,
                   '--plan',f'configs/{BATCH}.json','--root',f'/persist/{BATCH}/{arm}/output',
                   '--device','cuda:0' if gpu else 'cpu']
    args += [image, *command]
    cid = docker(*args).stdout.decode().strip()
    save(d / f'launch-{name}.json', dict(name=container, id=cid, argv=args,
         source_sha256=MANIFEST, actual_image=image, gpu=gpu, created_epoch=time.time(), automatic_retry=False))
    tar = ROOT / 'cross-server-handoff/frozen-source.tar'
    with tar.open('rb') as f:
        docker('cp', '-', f'{container}:/', input=f.read())
    save(d / f'source-deployment-{name}.json', dict(source_sha256=MANIFEST, archive_sha256=sha(tar)))
    docker('start', container)
    started = time.monotonic()
    while True:
        state = json.loads(docker('inspect','--format','{{json .State}}',container).stdout)
        status = dict(state='running', stage=name, container=container, gpu=gpu,
                      elapsed_seconds=time.monotonic()-started, utc_epoch=time.time())
        save(STATE / arm / 'status.json', status)
        save(d / 'controller-state.json', status)
        if not state['Running']:
            logs = docker('logs',container)
            (d / f'{name}.log').write_bytes(logs.stdout + logs.stderr)
            save(d / f'exit-{name}.json', state)
            save(d / f'duration-{name}.json', dict(elapsed_seconds=time.monotonic()-started))
            assert state['ExitCode'] == 0, f'Failed container retained: {container}'
            result = read(d / 'output' / ('result.json' if name == 'report' else f'{name}/result.json'))
            if name != 'report':
                assert result['passed'] is True
            return
        if time.monotonic()-started > timeout:
            docker('stop','--time','30',container)
            raise TimeoutError(f'Only this batch container stopped on timeout: {container}')
        time.sleep(10)


def zero_probe(arm, image, gpu):
    name = f'ce-ablation-sn8200-v2-{arm}-cuda-check'
    code = "import json,torch; torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; a=torch.randn(256,256,device='cuda',requires_grad=True); b=a@a.T; b.square().mean().backward(); torch.cuda.synchronize(); assert torch.isfinite(b).all() and torch.isfinite(a.grad).all(); print(json.dumps({'passed':True,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'optimizer_updates':0}))"
    docker('run','--detach','--name',name,'--label',f'lm.batch={BATCH}',
           '--user','1015:1016','--network','none','--cpus','4','--memory','8g',
           '--gpus','device='+gpu['uuid'],image,'python','-c',code)
    started = time.monotonic()
    while True:
        state = json.loads(docker('inspect','--format','{{json .State}}',name).stdout)
        if not state['Running']:
            logs=docker('logs',name)
            save(STATE/arm/'cuda-check.json',dict(gpu=gpu,container=name,state=state,
                 output=(logs.stdout+logs.stderr).decode(),passed=state['ExitCode']==0))
            assert state['ExitCode']==0, 'Zero-update CUDA check failed'
            return
        if time.monotonic()-started>180:
            docker('stop','--time','5',name)
            raise TimeoutError('Zero-update CUDA check timed out; no science launched')
        time.sleep(3)


def worker(arm, image, allowed):
    gpu, lock = None, None
    try:
        assert existing(arm,'gate')
        gpu, lock = lease(arm, allowed)
        save(STATE / arm / 'admission.json', dict(gpu=gpu, utc_epoch=time.time()))
        zero_probe(arm,image,gpu)
        for s in ('audit-gate','science','audit'):
            stage(arm,s,image,gpu)
        science = read(ROOT / arm / 'output/science/result.json')
        audit = read(ROOT / arm / 'output/audit/result.json')
        assert science['scientific_optimizer_updates'] == 2560
        assert audit['replayed_states'] == 129 and audit['scientific_optimizer_updates'] == 0
        with MUTEX:
            LEASED.remove(gpu['uuid'])
            lock.close()
            lock = None
        stage(arm,'cloud',image)
        gate = read(ROOT / arm / 'output/gate/result.json')
        completion = dict(state='complete', passed=True, eligible=True, arm=arm,
            architecture=arm.split('-')[0], completed_stages=['gate','audit-gate','science','audit','cloud'],
            scientific_updates=2560, new_foundation_updates=0,
            development_optimizer_updates=gate['development_optimizer_updates'],
            optimizer_updates=gate['development_optimizer_updates']+2560,
            independent_audit_passed=True, full_cloud_json_verified=True,
            source_sha256=MANIFEST, actual_image=image, gpu=gpu, automatic_retry=False)
        save(ROOT / arm / 'completion-manifest.json',completion)
        save(ROOT / arm / 'controller-state.json',completion)
        save(STATE / arm / 'status.json',completion)
        return completion
    except BaseException as exc:
        save(STATE / arm / 'status.json',dict(state='needs_attention',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if lock is not None:
            with MUTEX:
                LEASED.discard(gpu['uuid'])
                lock.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--candidate-gpus',required=True,help='Comma-separated candidate UUIDs; each receives a zero-update CUDA check')
    args=p.parse_args()
    STATE.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (STATE/'execution.lock').open('a') as lock, (ROOT/'controller.lock').open('a') as batch_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(batch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (STATE/'controller-process.json').exists(), 'No automatic controller resubmission'
        assert sha(ROOT/'source-v1/source-manifest.json')==MANIFEST
        assert sha(ROOT/'cross-server-handoff/frozen-source.tar')==read(ROOT/'cross-server-handoff/manifest.json')['source_tar_sha256']
        assert read(STATE/'preflight.json')['passed'] is True
        for arm in ARMS:
            assert not existing(arm,'science'), 'Original science must not have started'
        save(STATE/'controller-process.json',dict(pid=os.getpid(),host=os.uname().nodename,
             image=args.image,arms=ARMS,started_epoch=time.time(),automatic_retry=False))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(worker,a,args.image,set(args.candidate_gpus.split(','))):a for a in ARMS}
            errors={}
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except BaseException as exc:
                    errors[futures[f]]=repr(exc)
        if errors:
            save(STATE/'controller-state.json',dict(state='needs_attention',errors=errors))
            raise RuntimeError(errors)
        for arm in (*ARMS,'loop-lower-lr','loop-no-bound'):
            assert read(ROOT/arm/'completion-manifest.json')['passed'] is True
        stage('report','report',args.image,command=['python','scripts/report_ce_cross_sample_ablation.py',
              '--plan',f'configs/{BATCH}.json','--batch',f'/persist/{BATCH}',
              '--out',f'/persist/{BATCH}/report/output'])
        save(STATE/'controller-state.json',dict(state='complete',passed=True,scientific_updates=10240,
             independent_reload=True,cloud_verified=True,report=str(ROOT/'report/output/result.json')))

if __name__=='__main__':
    main()
