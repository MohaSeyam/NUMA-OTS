#!/usr/bin/env bash
# run_stage2_inside_guest.sh — 4-Node Virtual NUMA Guest Verification Script
#
# 1. Captures 4-Node Virtual NUMA Topology (numactl --hardware & lstopo --of txt).
# 2. Builds & runs 05_qemu_numa_functional_verify.c (PHHT + move_pages(2)).
# 3. Builds libnuma_ots_hook.so (00_libnuma_ots_hook.c) and runs the unmodified
#    OpenMP trace_extractor binary under LD_PRELOAD=./libnuma_ots_hook.so across
#    the 4 virtual NUMA nodes, saving hook_telemetry_qemu4n.csv.
set -euo pipefail
SHARED_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/numa_ots_guest_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
cp "$SHARED_DIR"/*.c "$SHARED_DIR"/*.h "$BUILD_DIR"/
cd "$BUILD_DIR"

echo "=== [Stage 2.1] Installing numactl, libnuma-dev, hwloc, and gcc inside guest ==="
sudo apt-get update
sudo apt-get install -y build-essential numactl libnuma-dev hwloc-nox

echo "=== [Stage 2.2] Capturing 4-Node Virtual NUMA Topology (numactl & lstopo) ==="
numactl --hardware | tee "$SHARED_DIR/qemu_numactl.txt"
lstopo-no-graphics --of txt | tee "$SHARED_DIR/qemu_lstopo.txt"

echo "=== [Stage 2.3] Compiling & Running 05_qemu_numa_functional_verify.c ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
    05_qemu_numa_functional_verify.c -o qemu_numa_verify -lnuma -lm

OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=spread \
    ./qemu_numa_verify "$SHARED_DIR/qemu_functional_verify.csv" "$SHARED_DIR/qemu_functional_report.txt"

echo "=== [Stage 2.4] Compiling & Testing LD_PRELOAD libnuma_ots_hook.so on 4-Node Guest ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared \
    00_libnuma_ots_hook.c -o libnuma_ots_hook.so -ldl -lpthread
gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
    01_trace_extractor_openmp.c -o trace_extractor -lm

NUMA_OTS_LOG="$SHARED_DIR/hook_telemetry_qemu4n.csv" OMP_NUM_THREADS=8 \
    LD_PRELOAD=./libnuma_ots_hook.so ./trace_extractor traces_out

cat "$SHARED_DIR/qemu_functional_report.txt"
echo "=== Stage 2 Functional & LD_PRELOAD Verification Complete ==="
