# Build only a new batch image; leave the base image and running containers intact.
ARG BASE_IMAGE=sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2
FROM ${BASE_IMAGE}
LABEL lm-memory.batch="gated-loop-depth-extrapolation-v1"
COPY requirements-plotting.txt /opt/gated-depth/requirements-plotting.txt
RUN python -m pip install --no-cache-dir --no-deps --require-hashes \
    --only-binary=:all: --index-url https://pypi.org/simple \
    -r /opt/gated-depth/requirements-plotting.txt
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/matplotlib
