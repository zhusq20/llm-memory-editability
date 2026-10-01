import json, os, signal, time
from pathlib import Path
folder=Path('/tmp/llm-memory-siqizhu4-mechanism-20260927')
record_path=folder/'p2-io-throttle.json'
record=json.loads(record_path.read_text())
paused={j['pid']:j for j in record['paused']}
runner=2475840
seen=set()
resumed=[]
while paused:
    try:
        children=Path(f'/proc/{runner}/task/{runner}/children').read_text().split()
    except FileNotFoundError:
        children=[]
    active={}
    for token in children:
        pid=int(token)
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'llm_memory_editability.bios_cross_continue' not in cmd: continue
            destination=cmd[cmd.index(b'--output')+1].decode()
            if '/llm-memory/results/bios-mechanism-dev-v1/p2/' not in destination: continue
            seen.add(destination)
            state=Path(f'/proc/{pid}/status').read_text().split('State:',1)[1].splitlines()[0].strip()[0]
            env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            gpu=int(next(v.split(b'=',1)[1] for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')))
            if pid not in paused and state not in ('T','t','Z'): active[gpu]=active.get(gpu,0)+1
        except (FileNotFoundError, ProcessLookupError):
            continue
    # Let the original runner dispatch all remaining planned jobs first. Afterwards,
    # resume parked workers into freed slots, never changing their in-memory state.
    ready=len(seen)>=36 or not children
    if ready:
        for pid,job in list(paused.items()):
            if active.get(job['gpu'],0)>=3: continue
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                if b'llm_memory_editability.bios_cross_continue' not in cmd:
                    raise RuntimeError('Parked worker identity changed')
                os.kill(pid,signal.SIGCONT)
                active[job['gpu']]=active.get(job['gpu'],0)+1
                resumed.append({**job,'resumed_at':time.time()})
            except (FileNotFoundError,ProcessLookupError):
                resumed.append({**job,'no_longer_alive_at':time.time()})
            del paused[pid]
    record.update(resumed=resumed,remaining_paused=list(paused.values()),seen_jobs=len(seen),resume_required=bool(paused),updated=time.time())
    tmp=record_path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(record_path)
    if paused: time.sleep(15)
