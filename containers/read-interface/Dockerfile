# Fixed LM image sha256:27b3ab67… (tag lm-memory-ce-ablation:20261017; (torch 2.8.0+cu128, ruff, wandb 0.30.0) with transformers new enough for Qwen3.5.
FROM lm-memory-ce-ablation:20261017
RUN pip install --no-cache-dir transformers==5.20.2 datasets==4.4.1
ENV PYTHONPATH=/project/src
WORKDIR /project
