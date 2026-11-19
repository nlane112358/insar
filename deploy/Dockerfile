# Build the shared C++ engine, thin CLI wrappers, and in-process Python binding.
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgdal-dev python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --no-cache-dir nanobind

WORKDIR /src
COPY CMakeLists.txt .
COPY cpp/ ./cpp/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DINSAR_BUILD_TESTS=OFF \
      -DINSAR_BUILD_PYTHON=ON \
      -DINSAR_NATIVE_ARCH=OFF \
      -DPython_EXECUTABLE=/opt/build-venv/bin/python \
      -DCMAKE_INSTALL_PREFIX=/opt/insar \
    && cmake --build build -j2 \
    && cmake --install build

# Shared Python/runtime base for the control API and worker images.
FROM ubuntu:24.04 AS runtime-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal34t64 libgomp1 libstdc++6 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.txt ./deploy/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r ./deploy/requirements.txt

COPY pyproject.toml ./
COPY python/ ./python/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/python" \
    INSAR_DATA_DIR="/data"

# The compute image contains libinsar; the API image below does not.
FROM runtime-base AS worker-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gdal python3-h5py python3-matplotlib python3-numpy \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements-ingest.txt ./deploy/requirements-ingest.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r ./deploy/requirements-ingest.txt
COPY --from=builder /opt/insar/ /opt/insar/
ENV LD_LIBRARY_PATH="/opt/insar/lib" \
    PYTHONPATH="/app/python:/opt/insar:/usr/lib/python3/dist-packages" \
    INSAR_ENGINE_BIN="/opt/insar/bin/insar_proto" \
    INSAR_STACK_BIN="/opt/insar/bin/insar_stack"

FROM worker-runtime AS worker
CMD ["celery", "-A", "insar_platform.workflows.celery_app:celery_app", "worker", "--loglevel=INFO", "--concurrency=1"]

FROM worker-runtime AS scheduler
CMD ["celery", "-A", "insar_platform.workflows.celery_app:celery_app", "beat", "--loglevel=INFO"]

# Keep the final stage as the API image for backwards-compatible local use.
FROM runtime-base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["uvicorn", "insar_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
