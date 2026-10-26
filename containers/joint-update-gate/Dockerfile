# Built by the local LM daemon, from the verified image already on this host.
ARG BASE_IMAGE=lm-memory-ce-ablation:20261017
FROM ${BASE_IMAGE}
LABEL lm-memory.batch="joint-update-gate-multihop-v1"
LABEL lm-memory.runtime-origin="local-build-from-pinned-lm-runtime"
COPY runtime-versions.json /opt/joint-update-gate/runtime-versions.json
RUN python -c 'import importlib.metadata as m,json; from pathlib import Path; expected=json.loads(Path("/opt/joint-update-gate/runtime-versions.json").read_text()); actual={name:m.version(name) for name in expected}; assert actual == expected, (actual, expected)'
