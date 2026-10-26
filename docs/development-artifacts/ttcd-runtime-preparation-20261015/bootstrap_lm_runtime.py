"""Host-specific independent LM daemon + NFS bind mount, explicitly authorized 2026-10-15.

Run as host root with --execute. It never restarts the default Docker service,
modifies its configuration, or runs experiments through its socket.
"""

import argparse
import grp
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

SOURCE = Path("/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace")
TARGET = Path("/ossfs/workspace")
CONFIG = Path("/etc/docker/lm-memory-daemon.json")
UNITS = Path("/etc/systemd/system")


def call(args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def write_new_or_identical(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Existing configuration differs; preserving {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)
    path.chmod(0o644)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source": str(SOURCE),
                    "target": str(TARGET),
                    "daemon_config": str(CONFIG),
                    "executed": False,
                }
            )
        )
        return
    if os.geteuid() != 0 or socket.gethostname() != "sn4622129202":
        raise RuntimeError("This recipe is restricted to host root on sn4622129202")
    os.umask(0o022)
    if os.readlink("/proc/self/ns/mnt") != os.readlink("/proc/1/ns/mnt"):
        raise RuntimeError("Enter the host mount namespace before running")
    fs = json.loads(call(["findmnt", "--json", "-T", str(SOURCE), "-o", "TARGET,SOURCE,FSTYPE"]))[
        "filesystems"
    ][0]
    if fs["fstype"] not in {"nfs", "nfs4"}:
        raise RuntimeError("Persistent source must be the observed server NFS mount")
    if not (SOURCE / "llm-memory-editability").is_dir():
        raise RuntimeError("Prepare the user-owned persistent source first")
    if shutil.disk_usage("/").free < 35 * 2**30:
        raise RuntimeError("Insufficient local image space; do not clean other projects")
    default_pid = call(["systemctl", "show", "docker", "--property=MainPID", "--value"])
    TARGET.mkdir(parents=True, exist_ok=True)
    TARGET.parent.chmod(0o755)
    if not TARGET.is_mount() and any(TARGET.iterdir()):
        raise RuntimeError("Refusing to cover existing contents at /ossfs/workspace")
    if TARGET.is_mount() and not os.path.samefile(SOURCE, TARGET):
        raise RuntimeError("A different persistent mount already occupies the target")
    group = grp.getgrnam("docker").gr_gid
    for path in (Path("/run/docker-lm-memory"), Path("/workspaces/docker-lm-memory")):
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, 0, group)
        path.chmod(0o750)
    daemon = {
        "hosts": ["unix:///run/docker-lm-memory/docker.sock"],
        "data-root": "/workspaces/docker-lm-memory",
        "exec-root": "/run/docker-lm-memory/exec",
        "pidfile": "/run/docker-lm-memory/docker.pid",
        "group": "docker",
        "storage-driver": "overlay2",
        "features": {"containerd-snapshotter": False},
        "containerd-namespace": "lm-memory",
        "containerd-plugins-namespace": "plugins.lm-memory",
        "bridge": "none",
        "iptables": False,
        "ip6tables": False,
        "ip-forward": False,
        "ip-masq": False,
        "runtimes": {"nvidia": {"path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": []}},
        "exec-opts": ["native.cgroupdriver=systemd"],
        "cgroup-parent": "lm-memory.slice",
        "live-restore": True,
        "max-concurrent-downloads": 2,
        "log-driver": "local",
        "log-opts": {"max-size": "10m", "max-file": "3"},
    }
    write_new_or_identical(CONFIG, json.dumps(daemon, indent=2) + "\n")
    call(["/usr/bin/dockerd", "--validate", "--config-file", str(CONFIG)])
    write_new_or_identical(
        UNITS / "ossfs-workspace.mount",
        f"""[Unit]
Description=LM persistent workspace backed by existing server NFS
DefaultDependencies=no
RequiresMountsFor={SOURCE}
After=network-online.target remote-fs.target
Wants=network-online.target
Conflicts=umount.target
Before=umount.target

[Mount]
What={SOURCE}
Where={TARGET}
Type=none
Options=bind
TimeoutSec=30

[Install]
WantedBy=multi-user.target
""",
    )
    write_new_or_identical(
        UNITS / "lm-memory.slice",
        """[Unit]
Description=Independent LM Memory Editability resource slice

[Slice]
CPUAccounting=yes
MemoryAccounting=yes
""",
    )
    write_new_or_identical(
        UNITS / "docker-lm-memory.service",
        """[Unit]
Description=Independent LM Memory Editability Docker daemon
Requires=ossfs-workspace.mount
After=network-online.target ossfs-workspace.mount
Wants=network-online.target

[Service]
Type=notify
Group=docker
ExecStart=/usr/bin/dockerd --config-file=/etc/docker/lm-memory-daemon.json
Restart=on-failure
RestartSec=5
TimeoutStartSec=60
TimeoutStopSec=60
RuntimeDirectory=docker-lm-memory
RuntimeDirectoryMode=0750
Delegate=yes
KillMode=process
Slice=lm-memory.slice
LimitNOFILE=1048576
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity

[Install]
WantedBy=multi-user.target
""",
    )
    call(["systemctl", "daemon-reload"])
    call(["systemctl", "enable", "--now", "ossfs-workspace.mount"])
    call(["systemctl", "enable", "--now", "docker-lm-memory.service"])
    if not os.path.samefile(SOURCE, TARGET):
        raise RuntimeError("Persistent bind identity check failed")
    if call(["systemctl", "show", "docker", "--property=MainPID", "--value"]) != default_pid:
        raise RuntimeError("Default Docker PID changed during setup; inspect independently")
    print(
        json.dumps(
            {
                "prepared": True,
                "persistent_source": fs,
                "target": str(TARGET),
                "default_docker_pid_unchanged": True,
                "default_docker_pid": default_pid,
                "daemon_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "lm_service": call(["systemctl", "is-active", "docker-lm-memory"]),
            }
        )
    )


if __name__ == "__main__":
    main()
