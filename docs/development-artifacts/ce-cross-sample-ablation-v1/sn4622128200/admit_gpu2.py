"""Adopt live containers unchanged and admit user-confirmed, CUDA-verified GPU2."""
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

spec=importlib.util.spec_from_file_location('continuation',Path(__file__).with_name('continue_four.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
GPU2='GPU-9c6ca0d7-11bb-e940-c845-efea3cc49c23'
IMAGE='sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2'
original_inventory=r.inventory
original_stage=r.stage
original_lease=r.lease
original_probe=r.zero_probe


def inventory():
    rows=original_inventory()
    query=subprocess.run(['nvidia-smi','-i',GPU2,'--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=20)
    idx,uid,used,util=[x.strip() for x in query.stdout.strip().split(',')]
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],
        check=True,capture_output=True,text=True,timeout=20)
    if int(used)<128 and uid not in processes.stdout.splitlines() and uid not in {x['uuid'] for x in rows}:
        rows.append(dict(index=int(idx),uuid=uid,memory_used=int(used),utilization=int(util),
                         admission='user-confirmed CUDA-verified; utilization alone is not exclusion'))
    return rows


def lease(arm,allowed):
    if arm=='loop-carry-adam':
        launch=r.read(r.ROOT/arm/'launch-science.json')
        actual=json.loads(r.docker('inspect',launch['name']).stdout)[0]
        assert actual['Id']==launch['id'] and actual['Image']==IMAGE
        assert actual['State']['Running'] or actual['State']['ExitCode']==0
        gpu=launch['gpu']
        with r.MUTEX:
            assert gpu['uuid'] not in r.LEASED
            lock=(r.STATE/('gpu-'+gpu['uuid']+'.lock')).open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            r.LEASED.add(gpu['uuid'])
        return gpu,lock
    return original_lease(arm,allowed)


def probe(arm,image,gpu):
    if arm=='loop-carry-adam':
        assert r.read(r.STATE/arm/'cuda-check.json')['passed']
        return
    if gpu['uuid']==GPU2:
        state=json.loads(r.docker('inspect','--format','{{json .State}}','ce-ablation-sn8200-gpu2-check').stdout)
        assert not state['Running'] and state['ExitCode']==0
        r.save(r.STATE/arm/'cuda-check.json',dict(passed=True,gpu=gpu,reused_zero_update_check='ce-ablation-sn8200-gpu2-check'))
        return
    return original_probe(arm,image,gpu)


def stage(arm,name,image,gpu=None,command=None,timeout=43200):
    d=r.ROOT/arm
    launch_path=d/f'launch-{name}.json'
    if launch_path.exists() and not (d/f'exit-{name}.json').exists():
        launch=r.read(launch_path)
        actual=json.loads(r.docker('inspect',launch['name']).stdout)[0]
        assert actual['Id']==launch['id'] and actual['Image']==image and launch['source_sha256']==r.MANIFEST
        assert actual['Config']['Labels']['lm.batch']==r.BATCH
        r.save(r.STATE/arm/f'adopted-{name}.json',dict(container=launch['name'],container_id=launch['id'],
            original_process_pid=actual['State']['Pid'],unchanged_running_container=True,utc_epoch=time.time()))
        started=time.monotonic()
        elapsed_before=time.time()-launch['created_epoch']
        while True:
            state=json.loads(r.docker('inspect','--format','{{json .State}}',launch['name']).stdout)
            elapsed=elapsed_before+time.monotonic()-started
            status=dict(state='running',stage=name,container=launch['name'],gpu=gpu,elapsed_seconds=elapsed,
                        adopted_without_restart=True,utc_epoch=time.time())
            r.save(r.STATE/arm/'status.json',status)
            r.save(d/'controller-state.json',status)
            if not state['Running']:
                logs=r.docker('logs',launch['name'])
                (d/f'{name}.log').write_bytes(logs.stdout+logs.stderr)
                r.save(d/f'exit-{name}.json',state)
                r.save(d/f'duration-{name}.json',dict(elapsed_seconds=elapsed,adopted_without_restart=True))
                assert state['ExitCode']==0
                assert r.read(d/'output'/name/'result.json')['passed']
                return
            if elapsed>timeout:
                r.docker('stop','--time','30',launch['name'])
                raise TimeoutError('Original stage timeout; only this batch container stopped')
            time.sleep(10)
    return original_stage(arm,name,image,gpu,command,timeout)


r.inventory=inventory
r.lease=lease
r.zero_probe=probe
r.stage=stage
with (r.STATE/'execution.lock').open('a') as lock,(r.ROOT/'controller.lock').open('a') as batch_lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(batch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (r.STATE/'controller-process.json').exists()
    r.save(r.STATE/'controller-process.json',dict(pid=os.getpid(),host=os.uname().nodename,image=IMAGE,
        arms=r.ARMS,started_epoch=time.time(),automatic_retry=False,adopts_live_containers=True))
    allowed={'GPU-364f0e00-7f5f-e6a7-9fea-03eb31c0bb3b',GPU2,
             'GPU-9dc24b81-029c-288b-ced5-ce32b996bae3','GPU-681ddfc6-46d1-0d48-ce5d-f9ec6cbf278c'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(r.worker,a,IMAGE,allowed):a for a in r.ARMS}
        errors={}
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except BaseException as exc:
                errors[futures[f]]=repr(exc)
    if errors:
        r.save(r.STATE/'controller-state.json',dict(state='needs_attention',errors=errors))
        raise RuntimeError(errors)
    for arm in (*r.ARMS,'loop-lower-lr','loop-no-bound'):
        assert r.read(r.ROOT/arm/'completion-manifest.json')['passed']
    stage('report','report',IMAGE,command=['python','scripts/report_ce_cross_sample_ablation.py',
        '--plan',f'configs/{r.BATCH}.json','--batch',f'/persist/{r.BATCH}',
        '--out',f'/persist/{r.BATCH}/report/output'])
    r.save(r.STATE/'controller-state.json',dict(state='complete',passed=True,scientific_updates=10240,
        independent_reload=True,cloud_verified=True,report=str(r.ROOT/'report/output/result.json')))
