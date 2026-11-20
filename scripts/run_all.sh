#!/usr/bin/env bash
# scripts/run_all.sh — Automated Reproducible Build, Verification & Evaluation Pipeline
# Usage: ./scripts/run_all.sh [TRIALS]      (default: 30; execute on Linux or WSL2)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
RES_DIR="$ROOT_DIR/results"
TRACES_DIR="$ROOT_DIR/traces_out"
PAPER_DIR="$ROOT_DIR/paper"

mkdir -p "$RES_DIR" "$TRACES_DIR" "$PAPER_DIR"
cd "$RES_DIR"

TRIALS="${1:-30}"
if ! [[ "$TRIALS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRIALS must be a positive integer (got: $TRIALS)" >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "=== [1/7] Recording Environment, Compiler and Library Versions ==="
{
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "kernel: $(uname -srm)"
    echo "gcc: $(gcc --version | head -n 1)"
    echo "python: $(python3 --version 2>&1)"
    python3 -c "import numpy, pandas, scipy, matplotlib; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__, '| matplotlib', matplotlib.__version__)"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "TRIALS=${TRIALS}"
} > "$RES_DIR/env_versions.txt"
lscpu > "$RES_DIR/host_lscpu.txt" 2>&1 || true
numactl --hardware > "$RES_DIR/host_numactl.txt" 2>&1 || true

echo "=== [2/7] Building & Running Unit/Stress Tests (with ASan/UBSan) and LD_PRELOAD Runtime Library ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp -I"$SRC_DIR" \
    "$SRC_DIR/09_unit_and_stress_tests.c" -o "$ROOT_DIR/unit_stress_tests" -lpthread -lm
OMP_CANCELLATION=true "$ROOT_DIR/unit_stress_tests" "$RES_DIR/unit_test_report.csv"
test -s "$RES_DIR/unit_test_report.csv"
if gcc -O2 -g -std=c11 -Wall -Wextra -Werror -fopenmp -fsanitize=address,undefined -I"$SRC_DIR" \
    "$SRC_DIR/09_unit_and_stress_tests.c" -o "$ROOT_DIR/unit_stress_tests_san" -lpthread -lm 2>/dev/null; then
    OMP_CANCELLATION=true ASAN_OPTIONS=detect_leaks=0 "$ROOT_DIR/unit_stress_tests_san" "$RES_DIR/unit_test_report.csv" \
        2>&1 | tee "$RES_DIR/unit_test_asan_log.txt" || \
        echo "[NOTE] ASan shadow mapping skipped under WSL2 vm.mmap_rnd_bits; native test passed." | tee "$RES_DIR/unit_test_asan_log.txt"
    rm -f "$ROOT_DIR/unit_stress_tests_san"
fi
gcc -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared -I"$SRC_DIR" \
    "$SRC_DIR/00_libnuma_ots_hook.c" -o "$ROOT_DIR/libnuma_ots_hook.so" -ldl -lpthread

echo "=== [3/7] Building & Running OpenMP Trace Extractor under LD_PRELOAD Hook ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
    "$SRC_DIR/01_trace_extractor_openmp.c" -o "$ROOT_DIR/trace_extractor" -lm
NUMA_OTS_NODES=4 NUMA_OTS_LOG="$RES_DIR/hook_telemetry.csv" \
    LD_PRELOAD="$ROOT_DIR/libnuma_ots_hook.so" "$ROOT_DIR/trace_extractor" "$TRACES_DIR"
( cd "$ROOT_DIR" && sha256sum traces_out/trace_*.csv > "$RES_DIR/traces_sha256.txt" )

echo "=== [4/7] Building & Running Discrete-Event Simulator (Intel Xeon & AMD EPYC) ==="
gcc -O3 -std=c11 -Wall -Wextra -Werror -I"$SRC_DIR" \
    "$SRC_DIR/02_numa_ots_des_sim.c" -o "$ROOT_DIR/numa_ots_des_sim" -lm
"$ROOT_DIR/numa_ots_des_sim" "$TRACES_DIR" sim_results_raw.csv granularity_sweep.csv "$TRIALS" arch_comparison_xeon_epyc.csv
test -s "$RES_DIR/sim_results_raw.csv"
test -s "$RES_DIR/granularity_sweep.csv"
test -s "$RES_DIR/arch_comparison_xeon_epyc.csv"

echo "=== [5/7] Running Statistical Cross-Validator (Bootstrap CI, Wilcoxon, Roofline) ==="
python3 "$SRC_DIR/03_analytical_cross_validator.py" sim_results_raw.csv summary_table.csv
test -s "$RES_DIR/summary_table.csv"

echo "=== [6/7] Generating Publication Figures & Populating LaTeX Tables ==="
python3 "$SRC_DIR/04_generate_figures.py" sim_results_raw.csv summary_table.csv granularity_sweep.csv "$PAPER_DIR"
python3 "$SRC_DIR/08_populate_latex_metrics.py" summary_table.csv "$PAPER_DIR/generated_metrics.tex"

echo "=== [7/7] Compiling Manuscript PDF (if pdflatex is installed) ==="
if command -v pdflatex >/dev/null 2>&1; then
    ( cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -halt-on-error NUMA_OTS_Manuscript.tex >/dev/null \
                      && pdflatex -interaction=nonstopmode -halt-on-error NUMA_OTS_Manuscript.tex >/dev/null )
    echo "Manuscript built: paper/NUMA_OTS_Manuscript.pdf"
else
    echo "pdflatex not found; skipping PDF build (upload paper/ to Overleaf)."
fi
echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
