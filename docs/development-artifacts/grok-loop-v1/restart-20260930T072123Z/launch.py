import datetime,json,subprocess
from pathlib import Path
root=Path('/mnt/disk2_from_server2/siqizhu4/llm-memory')
dest=Path('/mnt/disk2_from_server2/siqizhu4/llm-memory/docs/development-artifacts/grok-loop-v1/restart-20260930T072123Z')
command=['/mnt/disk2_from_server2/siqizhu4/llm-memory/.venv/bin/python', 'scripts/execute_grok_loop.py', '--config', 'configs/grok-loop-confirmation-v1.json', '--gpus', '2', '5', '--per-gpu', '2']
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'started_utc':now(),'command':command,'state':'running'}
(dest/'status.json').write_text(json.dumps(state,indent=2)+'\n')
with (dest/'training.log').open('x') as log:
    result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
state.update(state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_utc=now())
(dest/'status.json').write_text(json.dumps(state,indent=2)+'\n')
