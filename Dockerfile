# Stage 1: Build C++20 InSAR engine with GDAL
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt .
COPY src/ ./src/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Stage 2: Slim OEM API Runtime
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal34t64 libgomp1 python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/build/insar_proto /app/bin/insar_proto
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
ENV PATH="/opt/venv/bin:$PATH" \
    INSAR_ENGINE_BIN="/app/bin/insar_proto" \
    INSAR_DATA_DIR="/data"

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
