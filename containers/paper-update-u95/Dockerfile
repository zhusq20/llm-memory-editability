# Build a batch image from the original P runtime already installed on this host.
# Scientific source and data are mounted read-only by the frozen launcher.
# Verify this local tag is sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee before building.
FROM lm-memory-ttcd:torch2.8-cu128
LABEL lm-memory.batch="paper-update-gate-multihop-u95-v1"
LABEL lm-memory.runtime-origin="original-paper-baseline"
COPY runtime-versions.json /opt/paper-update-u95/runtime-versions.json
RUN python -c 'import importlib.metadata as m,json; from pathlib import Path; p=Path("/opt/paper-update-u95/runtime-versions.json"); expected=json.loads(p.read_text()); actual={name:m.version(name) for name in expected}; assert actual == expected, (actual, expected)'
