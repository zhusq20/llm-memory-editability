FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
RUN pip install --no-cache-dir transformers==4.57.3 numpy==2.2.6 pytest==9.1.1 ruff==0.14.0 wandb==0.30.0 huggingface-hub==0.36.0
ENV PYTHONPATH=/project/src
WORKDIR /project
