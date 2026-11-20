# Dockerfile — Reproducible Container Environment for NUMA-OTS
# Build: docker build -t numa-ots-suite .
# Run  : docker run --rm -v "$(pwd):/workspace" numa-ots-suite ./scripts/run_all.sh 30

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    numactl libnuma-dev hwloc-nox \
    valgrind \
    python3 python3-numpy python3-pandas python3-scipy python3-matplotlib \
    dos2unix bsdextrautils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN dos2unix scripts/*.sh src/*.py && chmod +x scripts/*.sh src/*.py

CMD ["./scripts/run_all.sh", "30"]
