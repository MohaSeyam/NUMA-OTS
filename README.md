# NUMA-OTS: NUMA-Aware Task Scheduling for Shared-Memory Parallel Applications Using OpenMP

**Authors:** Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam  
**Affiliation:** Computer Engineering Department, Faculty of Engineering, Islamic University of Gaza

This repository contains the source code, `LD_PRELOAD` runtime interposition prototype, 8-test verification suite, discrete-event simulator, Valgrind Cachegrind micro-kernels, optional QEMU virtual-NUMA scripts, raw experimental datasets, and LaTeX manuscript for **NUMA-OTS** (*NUMA-Aware OpenMP Task Scheduler*).

---

## Repository Structure

```text
NUMA-OTS/
├── paper/          # IEEE-style LaTeX source, vector figures, auto-generated macros & final PDF
├── results/        # Audited statistical CSVs, raw 30-trial Xeon/EPYC datasets, test & host logs
├── scripts/        # Reproducibility drivers (Stage 1, optional Stage 2 QEMU, Stage 3 Cachegrind)
├── sniper_out/     # Raw Valgrind Cachegrind logs across SpMV, BFS, HashJoin, PageRank
├── src/            # C11 runtime hook, DES simulator, trace extractor, tests, Python analysis/plots
├── traces_out/     # Deterministic synthetic logical-page traces
├── CITATION.cff    # Citation metadata (GitHub "Cite this repository")
├── Dockerfile      # Reproducible Ubuntu container recipe
├── LICENSE         # MIT License
└── README.md
```

### Key Files

| Path | Description |
|---|---|
| `src/numa_ots_core.h` | Shared C11 header implementing GCC 13.2–15.2 `depend` ABI parsing, M1 Eq. (5) scoring/tie-breaking, M2 EMA step-change guard, and Proposition 1 $R_{\min}(B)$. |
| `src/00_libnuma_ots_hook.c` | Live `LD_PRELOAD` runtime prototype (`libnuma_ots_hook.so`) intercepting `GOMP_task` and `GOMP_taskwait`, maintaining the 64-stripe PHHT, and logging both submitted (`m2_mig_requests`) and kernel-confirmed (`m2_mig_confirmed`) `move_pages(2)` requests. |
| `src/01_trace_extractor_openmp.c` | OpenMP 5.2 smoke tests and deterministic synthetic logical-page trace extractor for SpMV, BFS, HashJoin, and PageRank. |
| `src/02_numa_ots_des_sim.c` | Pure-C11 min-heap discrete-event simulator evaluating **13 policies** (including reactive `AutoNUMA` and full M1/M2/M3 ablations) across 2, 4, and 8 NUMA domains for Intel Xeon UPI and AMD EPYC NPS4. |
| `src/03_analytical_cross_validator.py` | Computes 10,000-resample paired BCa bootstrap 95% CIs, Holm–Bonferroni Wilcoxon tests, Cohen's $d_z$, internal Roofline consistency checks, and the 13-policy ablation summary. |
| `src/04_generate_figures.py` | Generates vector PDF figures (`paper/fig1_architecture.pdf`, `paper/fig2_evaluation.pdf`, `paper/fig3_sensitivity.pdf`) with embedded fonts. |
| `src/05_qemu_numa_functional_verify.c` | Optional Linux `libnuma`/`move_pages(2)` functional verifier for 4-node virtual NUMA QEMU guests. |
| `src/06_sniper_microbench_openmp.c` | Controlled OpenMP micro-kernels for Stage 3 Valgrind Cachegrind temporal reuse and model-based energy analysis (`src/07_parse_sniper_results.py`). |
| `src/08_populate_latex_metrics.py` | Automatically synchronizes all LaTeX tables and inline numeric macros (`paper/generated_metrics.tex`) from `results/*.csv`. |
| `src/09_unit_and_stress_tests.c` | 8-test automated verification suite (`T1`–`T8`) covering PHHT concurrency, GCC ABI, M1/M2 math, OpenMP 5.2 `depend`/`untied`/`detach`/`cancel`, nested teams, and liveness (verified under ASan/UBSan). |
| `scripts/run_all.sh` | One-command reproducible driver executing Stages 1a–1c, generating all CSVs in `results/`, vector figures, LaTeX tables, and `paper/NUMA_OTS_Manuscript.pdf`. |
| `results/` | Contains `sim_results_raw.csv` (4,680 Xeon trials), `sim_results_epyc_raw.csv` (3,120 EPYC trials), `summary_table.csv`, `ablation_table.csv`, `controller_accounting.csv`, `migration_batch_hist.csv`, `granularity_sweep.csv`, `migration_sensitivity_sweep.csv`, `sniper_cache_energy.csv`, `hook_telemetry.csv`, `unit_test_report.csv`, `unit_test_asan_log.txt`, `traces_sha256.txt`, `env_versions.txt`, and `host_lscpu.txt`. |
| `paper/NUMA_OTS_Manuscript.pdf` | Compiled 10-page IEEE-style manuscript PDF (with LaTeX source in `paper/NUMA_OTS_Manuscript.tex`). |

---

## Reproducing the Experiments (Ubuntu / WSL2)

### 1. Main Pipeline (Unit Tests, `LD_PRELOAD` Prototype, Trace Extractor, 13-Policy DES, Statistics, Figures & PDF)

```bash
git clone https://github.com/MohaSeyam/NUMA-OTS.git
cd NUMA-OTS
sudo apt-get update
sudo apt-get install -y build-essential numactl python3 python3-numpy python3-pandas python3-scipy python3-matplotlib
chmod +x scripts/*.sh
./scripts/run_all.sh 30
```

### 2. Stage 3: Valgrind Cachegrind Micro-Benchmarks

```bash
./scripts/run_stage3_sniper.sh
```

### 3. Optional Stage 2: 4-Node Virtual NUMA Guest Verification (QEMU)

```bash
./scripts/run_stage2_qemu_host.sh
```

Inside the booted guest (`ubuntu` / `ubuntu`):

```bash
sudo mount -t 9p -o trans=virtio hostshare /mnt || true
cd /mnt && ./scripts/run_stage2_inside_guest.sh
```
