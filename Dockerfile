# Stage 1: Build C++20 InSAR engine (`insar_proto`) and sequential stacker (`insar_stack`) with GDAL + OpenMP
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt .
COPY include/ ./include/
COPY src/ ./src/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DINSAR_BUILD_TESTS=OFF \
    && cmake --build build --target insar_proto insar_stack -j

# Stage 2: Slim OEM API & Stack Runtime
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal34t64 libgomp1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/build/insar_proto /app/bin/insar_proto
COPY --from=builder /src/build/insar_stack /app/bin/insar_stack
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY scripts/ ./scripts/

ENV PATH="/app/bin:/opt/venv/bin:$PATH" \
    INSAR_ENGINE_BIN="/app/bin/insar_proto" \
    INSAR_STACK_BIN="/app/bin/insar_stack" \
    INSAR_DATA_DIR="/data"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
