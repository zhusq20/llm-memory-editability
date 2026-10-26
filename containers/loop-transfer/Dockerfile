FROM lm-memory-paper-update-u95:20261017-sn4622129202
LABEL lm-memory.batch="loop-queue-target-20261021"
COPY runtime-versions.json /opt/loop-transfer/runtime-versions.json
RUN python -c 'import importlib.metadata as m,json; from pathlib import Path; expected=json.loads(Path("/opt/loop-transfer/runtime-versions.json").read_text()); actual={k:m.version(k) for k in expected}; assert actual==expected,(actual,expected)'
ENV PYTHONPATH=/project/src:/project/scripts
WORKDIR /project
