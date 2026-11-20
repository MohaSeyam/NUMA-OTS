# NUMA-OTS: NUMA-Aware Task Scheduling for Shared-Memory Parallel Applications Using OpenMP

**Authors:** Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam  
**Affiliation:** Computer Engineering Department, Faculty of Engineering, Islamic University of Gaza

This repository contains the source code, `LD_PRELOAD` runtime interposition prototype, 8-test verification suite, discrete-event simulator, Valgrind Cachegrind micro-kernels, optional QEMU virtual-NUMA scripts, raw experimental datasets, and LaTeX manuscript for **NUMA-OTS** (*NUMA-Aware OpenMP Task Scheduler*).

---

## Repository Structure

| Path | Description |
|---|---|
| `numa_ots_core.h` | Shared C11 header implementing GCC 13.2–15.2 `depend` ABI parsing, M1 Eq. (5) scoring/tie-breaking, M2 EMA step-change guard, and Proposition 1 $R_{\min}(B)$. |
| `00_libnuma_ots_hook.c` | Live `LD_PRELOAD` runtime prototype (`libnuma_ots_hook.so`) intercepting `GOMP_task` and `GOMP_taskwait`, maintaining the 64-stripe PHHT, and logging both submitted and kernel-confirmed `move_pages(2)` requests. |
| `01_trace_extractor_openmp.c` | OpenMP 5.2 smoke tests and deterministic synthetic logical-page trace extractor for SpMV, BFS, HashJoin, and PageRank. |
| `02_numa_ots_des_sim.c` | Pure-C11 min-heap discrete-event simulator evaluating 13 policies (including reactive `AutoNUMA` and full M1/M2/M3 ablations) across 2, 4, and 8 NUMA domains for Intel Xeon UPI and AMD EPYC NPS4. |
| `03_analytical_cross_validator.py` | Computes 10,000-resample paired BCa bootstrap 95% CIs, Holm–Bonferroni Wilcoxon tests, Cohen's $d_z$, internal Roofline consistency checks, and the 13-policy ablation summary. |
| `04_generate_figures.py` | Generates vector PDF figures (`paper/fig1_architecture.pdf`, `paper/fig2_evaluation.pdf`, `paper/fig3_sensitivity.pdf`) with embedded fonts. |
| `05_qemu_numa_functional_verify.c` | Optional Linux `libnuma`/`move_pages(2)` functional verifier for 4-node virtual NUMA QEMU guests (`run_stage2_qemu_host.sh`, `run_stage2_inside_guest.sh`). |
| `06_sniper_microbench_openmp.c` | Controlled OpenMP micro-kernels for Stage 3 Valgrind Cachegrind temporal reuse and model-based energy analysis (`run_stage3_sniper.sh`, `07_parse_sniper_results.py`). |
| `08_populate_latex_metrics.py` | Automatically synchronizes all LaTeX tables and inline numeric macros (`paper/generated_metrics.tex`) from generated CSV outputs. |
| `09_unit_and_stress_tests.c` | 8-test automated verification suite (`T1`–`T8`) covering PHHT concurrency, GCC ABI, M1/M2 math, OpenMP 5.2 `depend`/`untied`/`detach`/`cancel`, nested teams, and liveness (verified under ASan/UBSan). |
| `run_all.sh` | One-command reproducible driver executing Stages 1a–1c, generating all CSVs, figures, LaTeX tables, and compiling `paper/NUMA_OTS_Manuscript.pdf`. |
| `traces_out/`, `traces_sha256.txt` | Deterministic logical-page traces and their SHA-256 checksums. |
| `sim_results_raw.csv`, `sim_results_epyc_raw.csv` | Complete raw 30-trial simulation datasets for Intel Xeon (4,680 rows) and AMD EPYC (3,120 rows). |
| `summary_table.csv`, `ablation_table.csv`, `controller_accounting.csv`, `migration_batch_hist.csv`, `granularity_sweep.csv`, `migration_sensitivity_sweep.csv`, `sniper_cache_energy.csv`, `hook_telemetry.csv` | Audited statistical summaries, controller accounting, migration batch histograms, sensitivity sweeps, Cachegrind outputs, and live hook telemetry. |
| `unit_test_report.csv`, `unit_test_asan_log.txt`, `env_versions.txt`, `host_lscpu.txt` | Automated test pass records (native + ASan/UBSan) and recorded host environment versions. |
| `paper/NUMA_OTS_Manuscript.tex`, `paper/NUMA_OTS_Manuscript.pdf` | Complete IEEE-style LaTeX source, auto-generated metric macros, vector figures, and compiled manuscript PDF. |
| `Dockerfile`, `CITATION.cff`, `LICENSE` | Container recipe, citation metadata (GitHub "Cite this repository"), and MIT License. |

---

## Architecture & Evaluation Pipeline

1. **Runtime Interposition Prototype & Automated Tests (`00_libnuma_ots_hook.c`, `09_unit_and_stress_tests.c`)**:
   - `libnuma_ots_hook.so` intercepts `GOMP_task` and `GOMP_taskwait` in `libgomp.so.1` via `LD_PRELOAD`, parses `/sys/devices/system/node/node<N>/cpulist`, resolves `depend` clause base pages into a 64-stripe lock-padded Page-Home Hash Table (PHHT) with epoch TTL via batched `move_pages(2)` queries, computes M1 affinity scores/priority hints, and submits M2 page-migration requests (`MPOL_MF_MOVE`), logging both submitted requests (`m2_mig_requests`) and kernel-confirmed cross-node moves (`m2_mig_confirmed`).
   - `09_unit_and_stress_tests.c` runs 8 automated tests (native + `-fsanitize=address,undefined` logged to `unit_test_asan_log.txt`) covering PHHT concurrency, GCC 13.2–15.2 `depend` ABI layouts, M1 Eq. (5), M2 Proposition 1 $R_{\min}(B)$, OpenMP 5.2 core constructs, `untied`/`detach` + `omp_fulfill_event`, nested teams / `cancel taskgroup`, and task-tree liveness.
2. **Stage 1 — Deterministic Trace Extraction & Multi-Socket DES (`01_trace_extractor_openmp.c`, `02_numa_ots_des_sim.c`, `03_analytical_cross_validator.py`)**:
   - Generates deterministic synthetic logical-page footprint traces (`1,542`–`1,932` unique 4 KiB logical pages per workload) across **CSR SpMV**, **Level-Synchronous Frontier BFS**, **Morsel-Driven Hash Join**, and **Iterative PageRank**.
   - Evaluates **13 policies** (`Base`, `Mem0`, `Interleave`, `ParFT`, reactive `AutoNUMA`, `M1`, `M2`, `M3`, `M1+M2`, `M1+M3`, `M2+M3`, `Ungated M2`, `Full NUMA-OTS`) across 2, 4, and 8 NUMA domains on calibrated **Intel Xeon (Cooper Lake UPI)** and **AMD EPYC (NPS4 Infinity Fabric)** topologies (30 paired trials per configuration, raw trials saved to `sim_results_raw.csv` and `sim_results_epyc_raw.csv`), plus 30-trial RQ1 granularity sweeps and RQ2 parameter sensitivity sweeps.
3. **Stage 2 — Optional 4-Node Virtual NUMA Functional Verification in QEMU (`05_qemu_numa_functional_verify.c`, `run_stage2_qemu_host.sh`, `run_stage2_inside_guest.sh`)**:
   - Optional host/guest scripts to boot an Ubuntu 24.04 Linux guest with 4 virtual NUMA domains (`ACPI SLIT`: `10`/`18`/`22`) and functionally verify `libnuma`, `hwloc`, and kernel `move_pages(2)` physical relocation (requires QEMU + Ubuntu cloud image; not pre-executed in WSL2-only runs).
4. **Stage 3 — Single-Hierarchy Cache Reuse & Model-Based Energy Micro-Benchmarks (`06_sniper_microbench_openmp.c`, `run_stage3_sniper.sh`, `07_parse_sniper_results.py`)**:
   - Runs controlled micro-kernels under **Valgrind Cachegrind** (`--cache-sim=yes`, scaled `16 KiB D1` and `64 KiB LL`) comparing fixed-partition reuse (`ots`) against cyclic rotation (`base`), reporting `D1` and simulated `LL` miss rates plus model-based energy estimates ($E = 0.42\,I_r + 1.8\,D_{1m} + 35\,\mathrm{LL}_m$ nJ).

---

## Reproducing the Experiments (Ubuntu / WSL2)

### 1. Main Pipeline (Unit Tests, `LD_PRELOAD` Prototype, Trace Extractor, 13-Policy DES, Statistics, Figures & PDF)

```bash
sudo apt-get update
sudo apt-get install -y build-essential numactl python3 python3-numpy python3-pandas python3-scipy python3-matplotlib
chmod +x run_all.sh
./run_all.sh 30
```

Generated outputs:
- `unit_test_report.csv`, `unit_test_asan_log.txt`, and `hook_telemetry.csv`
- `traces_out/trace_{spmv,bfs,join,pagerank}.csv` and `traces_sha256.txt`
- `sim_results_raw.csv`, `sim_results_epyc_raw.csv`, `ablation_table.csv`, `controller_accounting.csv`, `migration_batch_hist.csv`, `granularity_sweep.csv`, `migration_sensitivity_sweep.csv`, `arch_comparison_xeon_epyc.csv`, and `summary_table.csv`
- `paper/fig1_architecture.pdf`, `paper/fig2_evaluation.pdf`, `paper/fig3_sensitivity.pdf`
- `paper/generated_metrics.tex` and `paper/NUMA_OTS_Manuscript.pdf`

### 2. Stage 3: Valgrind Cachegrind Micro-Benchmarks

```bash
chmod +x run_stage3_sniper.sh
./run_stage3_sniper.sh
```

Generated output: `sniper_cache_energy.csv` (and raw Cachegrind logs under `sniper_out/`).

### 3. Optional Stage 2: 4-Node Virtual NUMA Guest Verification (QEMU)

```bash
chmod +x run_stage2_qemu_host.sh run_stage2_inside_guest.sh
./run_stage2_qemu_host.sh
```

Inside the booted guest (`ubuntu` / `ubuntu`):

```bash
sudo mount -t 9p -o trans=virtio hostshare /mnt || true
cd /mnt && ./run_stage2_inside_guest.sh
```
