"""Create only the dedicated LM runtime on the explicitly selected host."""
import json
import fcntl
import os
from pathlib import Path
import socket
import subprocess

BACKING = Path('/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/runtime-sn4622128200/docker-ext4.img')
MOUNT = Path('/workspaces/docker-lm-memory')
CONFIG = Path('/etc/docker/lm-memory-daemon.json')

def call(args):
    return subprocess.check_output(args, text=True).strip()

def new(path, value):
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError(f'Refusing to replace existing {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        path.chmod(0o644)

assert socket.gethostname() == 'sn4622128200' and os.geteuid() == 0
assert os.readlink('/proc/self/ns/mnt') == os.readlink('/proc/1/ns/mnt')
os.setegid(1016)
os.seteuid(1015)
assert BACKING.is_file() and BACKING.stat().st_uid == 1015 and BACKING.stat().st_size == 64 * 2**30
backing_fd = os.open(BACKING, os.O_RDWR)
os.seteuid(0)
os.setegid(0)
before = call(['systemctl','show','docker','--property=MainPID','--value'])
MOUNT.mkdir(parents=True, exist_ok=True)
if not MOUNT.is_mount():
    assert not list(MOUNT.iterdir()), 'Existing storage must be preserved'
    kind = subprocess.run(['blkid','-p','-s','TYPE','-o','value',str(BACKING)], capture_output=True,text=True,user=1015,group=1016)
    if kind.returncode == 2 and not kind.stdout.strip():
        subprocess.run(['mkfs.ext4','-F','-E','nodiscard,lazy_itable_init=1,lazy_journal_init=1',str(BACKING)],check=True,user=1015,group=1016)
    else:
        assert kind.stdout.strip() == 'ext4'
    control_fd = os.open('/dev/loop-control', os.O_RDWR)
    loop_number = fcntl.ioctl(control_fd, 0x4C82)
    loop_device = f'/dev/loop{loop_number}'
    loop_fd = os.open(loop_device, os.O_RDWR)
    fcntl.ioctl(loop_fd, 0x4C00, backing_fd)
    print('Dedicated LM storage device:', loop_device, flush=True)
    subprocess.run(['mount','-o','noatime',loop_device,str(MOUNT)],check=True)
    os.close(loop_fd)
    os.close(control_fd)
assert call(['findmnt','-n','-o','FSTYPE','--target',str(MOUNT)]) == 'ext4'
config = {
 'hosts':['unix:///run/docker-lm-memory/docker.sock'], 'data-root':str(MOUNT),
 'exec-root':'/run/docker-lm-memory/exec','pidfile':'/run/docker-lm-memory/docker.pid',
 'group':'siqizhu4','storage-driver':'overlay2',
 'containerd-namespace':'lm-memory','containerd-plugins-namespace':'plugins.lm-memory',
 'bridge':'none','iptables':False,'ip6tables':False,'ip-forward':False,'ip-masq':False,
 'runtimes':{'nvidia':{'path':'/usr/bin/nvidia-container-runtime','runtimeArgs':[]}},
 'exec-opts':['native.cgroupdriver=systemd'],'live-restore':True,
 'max-concurrent-downloads':2,'log-driver':'local','log-opts':{'max-size':'10m','max-file':'3'}
}
new(CONFIG,json.dumps(config,indent=2)+'\n')
subprocess.run(['/usr/bin/dockerd','--validate','--config-file',str(CONFIG)],check=True)
new(Path('/etc/systemd/system/docker-lm-memory.service'),'''[Unit]
Description=Dedicated LM experiment Docker runtime on sn4622128200
After=network-online.target
RequiresMountsFor=/workspaces/docker-lm-memory
[Service]
Type=notify
ExecStart=/usr/bin/dockerd --config-file=/etc/docker/lm-memory-daemon.json
Restart=on-failure
RestartSec=5
TimeoutStartSec=90
TimeoutStopSec=60
RuntimeDirectory=docker-lm-memory
RuntimeDirectoryMode=0755
Delegate=yes
KillMode=process
LimitNOFILE=1048576
TasksMax=infinity
''')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start','docker-lm-memory.service'],check=True)
assert call(['systemctl','show','docker','--property=MainPID','--value']) == before
print(json.dumps({'host':socket.gethostname(),'backing':str(BACKING),'mount':str(MOUNT),'default_docker_pid_unchanged':True,'auto_start_after_reboot':False}))
