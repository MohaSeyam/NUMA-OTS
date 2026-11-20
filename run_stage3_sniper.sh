#!/usr/bin/env bash
# run_stage3_sniper.sh — Cache Hierarchy Miss Rate & Energy Evaluation Script
#
# Runs the four micro-benchmarks (spmv, bfs, join, pagerank) across {base, ots}:
#   1. Executes Valgrind Cachegrind (--I1=16384,4,64 --D1=16384,4,64 --LL=65536,8,64).
#   2. Executes Sniper ($SNIPER_ROOT/run-sniper -n 8) + McPAT if Sniper is installed.
#   3. Parses results via 07_parse_sniper_results.py into sniper_cache_energy.csv.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [Stage 3.1] Compiling 06_sniper_microbench_openmp.c ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
    06_sniper_microbench_openmp.c -o sniper_microbench -lm

mkdir -p sniper_out

SNIPER_ROOT="${SNIPER_ROOT:-$HOME/snipersim}"
HAS_SNIPER=0
if [ -x "$SNIPER_ROOT/run-sniper" ]; then
    HAS_SNIPER=1
fi

HAS_VALGRIND=0
if command -v valgrind >/dev/null 2>&1; then
    HAS_VALGRIND=1
fi

if [ "$HAS_VALGRIND" -eq 0 ] && [ "$HAS_SNIPER" -eq 0 ]; then
    echo "Installing valgrind for instruction/cache hierarchy simulation..."
    sudo apt-get update && sudo apt-get install -y valgrind
    HAS_VALGRIND=1
fi

for wl in spmv bfs join pagerank; do
    for pol in base ots; do
        outdir="sniper_out/${wl}_${pol}"
        mkdir -p "$outdir"

        if [ "$HAS_VALGRIND" -eq 1 ]; then
            echo "=== [Stage 3.2a] Running Valgrind Cachegrind for workload=${wl}, policy=${pol} ==="
            # Cache sizes scaled proportionally to the micro-kernel working set (16 KiB D1, 64 KiB LL slice)
            # Note: Valgrind 3.22+ requires explicit --cache-sim=yes to enable cache hierarchy simulation.
            OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=8 valgrind --tool=cachegrind \
                --cache-sim=yes \
                --I1=16384,4,64 --D1=16384,4,64 --LL=65536,8,64 \
                --cachegrind-out-file="$outdir/cachegrind.out" \
                ./sniper_microbench "$wl" "$pol" > "$outdir/cg_stdout.txt" 2> "$outdir/cg_stderr.txt"
        fi

        if [ "$HAS_SNIPER" -eq 1 ]; then
            echo "=== [Stage 3.2b] Running Sniper (-n 8) for workload=${wl}, policy=${pol} ==="
            OMP_NUM_THREADS=8 "$SNIPER_ROOT/run-sniper" \
                -n 8 -c gainestown \
                -d "$outdir" \
                -- ./sniper_microbench "$wl" "$pol"
            if [ -x "$SNIPER_ROOT/tools/mcpat.py" ]; then
                python3 "$SNIPER_ROOT/tools/mcpat.py" -d "$outdir" -t total -o "$outdir/power" \
                    > "$outdir/powerstack.txt" 2>&1 || true
            fi
        fi
    done
done

echo "=== [Stage 3.3] Parsing Cache Miss Rates & Energy into sniper_cache_energy.csv ==="
python3 07_parse_sniper_results.py sniper_out sniper_cache_energy.csv
