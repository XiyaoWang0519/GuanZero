# PyTorch CUDA wheels include user-space CUDA libraries; the node supplies the
# NVIDIA driver and NVIDIA Container Toolkit. Build from the repository root.
FROM python:3.11-slim-bookworm
ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/python:/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build rsync openssh-client tini \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-dev.txt requirements-train.txt ./
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements-dev.txt -r requirements-train.txt
# Explicit COPY avoids transferring local runs, virtualenvs, and credential files.
COPY CMakeLists.txt ./
COPY cpp ./cpp
COPY python ./python
COPY train ./train
COPY eval ./eval
COPY oracle ./oracle
COPY tests ./tests
COPY scripts ./scripts
COPY infra ./infra
RUN find python -name '*.so' -delete \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DPython_EXECUTABLE=/usr/local/bin/python \
    && cmake --build build -j 4 \
    && ./build/cpp/tests/gd_tests \
    && python -c 'import torch, gd; print(torch.__version__, gd.OBS_DIM, gd.ACT_DIM)'
ENV PY=python
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./infra/run_train.sh"]
