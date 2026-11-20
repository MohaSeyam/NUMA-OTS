/*
 * 02_numa_ots_des_sim.c — Trace-Driven Multi-Socket NUMA Discrete-Event Simulator
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Description:
 *   Pure-C11 min-heap discrete-event simulator evaluating NUMA-OTS across 2, 4,
 *   and 8 NUMA domains for calibrated Intel Xeon (Cooper Lake UPI) and AMD EPYC
 *   (NPS4 Infinity Fabric) topologies across SpMV, BFS, HashJoin, and PageRank:
 *     - Exact undiscounted critical-path synchronous move_pages(2) barrier pause:
 *         T_mig_epoch = C_batch_fixed(N) + B_epoch * C_copy  (B_epoch <= 64)
 *     - Physically closed M/M/1 DRAM controller queueing model with exported
 *       per-controller accounting (controller_accounting.csv).
 *     - Non-tautological 6-epoch re-migration & ping-pong revisit tracking plus
 *       beneficial migration precision (% of migrations reducing remote misses).
 *     - Complete 13-policy comparison including AutoNUMA baseline and full
 *       single/pairwise component ablation (M1, M2, M3, M1+M2, M1+M3, M2+M3,
 *       Ungated M2, Full NUMA-OTS).
 *     - Separate static leakage vs. dynamic access/interconnect energy accounting.
 *     - 30-trial granularity sweep and mandatory parameter sensitivity sweeps
 *       (gamma, rem/loc ratio, misses/page, f_mem, node0_skew, gate ablation).
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -Werror 02_numa_ots_des_sim.c -o numa_ots_des_sim -lm
 * Run:
 *   ./numa_ots_des_sim traces_out sim_results_raw.csv granularity_sweep.csv 30 arch_comparison_xeon_epyc.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "numa_ots_core.h"

#define MAX_NODES      8
#define CORES_PER_NODE 4
#define MAX_CORES      (MAX_NODES * CORES_PER_NODE)
#define MAX_TOUCHED    16
#define HASH_TABLE_SZ  16384
#define HASH_MASK      (HASH_TABLE_SZ - 1)
#define DEQUE_CAP      2048
#define LLC_SETS       4096
#define LLC_MASK       (LLC_SETS - 1)
#define MAX_BATCH_MIG  64
#define PHASE_SWEEPS   10   /* Inner sweeps per phase -> realistic multi-ms iterative epochs */

typedef struct {
    int task_id;
    int phase;
    int creator_tid;
    int work_ops;
    int num_deps;
    uint32_t dep_pages[2];
    int dep_is_write[2];
    int num_pages;
    uint32_t pages[MAX_TOUCHED];
} trace_task_t;

typedef struct {
    uint32_t key;
    int used;
    int home_node;
    int phht_cached_node;
    int last_mig_phase;
    uint8_t visited_mask;      /* Bitmask of domains this page has resided on */
    int pending_eval;          /* 1 if migrated at end of previous phase */
    int prev_home;
    double prev_remote_misses;
    double cur_remote_misses;
    double dep_ready_ns;
    double R_hat[MAX_NODES];
    double R_obs[MAX_NODES];
} page_state_t;

typedef enum {
    ARCH_INTEL_XEON = 0,
    ARCH_AMD_EPYC   = 1
} arch_type_t;

typedef struct {
    arch_type_t arch;
    const char *arch_name;
    int n_nodes;
    double t_loc_ns;
    double t_rem_1hop_ns;
    double t_rem_2hop_ns;
    double t_rem_mean_ns;
    double c_copy_ns;
    double c_batch_fixed_ns;
    double c_hat_b64_ns;
    double ctrl_bw_cap;        /* Misses/phase at controller saturation (rho -> 1) */
    double L[MAX_NODES][MAX_NODES];
    int slit_order[MAX_NODES][MAX_NODES];
} numa_topo_t;

typedef enum {
    POL_UNGUIDED_BASE = 0,
    POL_PARFT,
    POL_MEMBIND0,
    POL_INTERLEAVE,
    POL_AUTONUMA,
    POL_OTS_M1_ONLY,
    POL_OTS_M2_ONLY,
    POL_OTS_M3_ONLY,
    POL_OTS_M1_M2,
    POL_OTS_M1_M3,
    POL_OTS_M2_M3,
    POL_OTS_UNGATED_M2,
    POL_OTS_FULL,
    POL_COUNT
} policy_t;

static const char *POLICY_NAMES[POL_COUNT] = {
    "sim_unguided_base", "parft", "membind0", "interleave", "autonuma",
    "numa_ots_m1_only", "numa_ots_m2_only", "numa_ots_m3_only",
    "numa_ots_m1_m2", "numa_ots_m1_m3", "numa_ots_m2_m3",
    "numa_ots_ungated_m2", "numa_ots_full"
};

static inline int pol_has_m1(policy_t p) {
    return (p == POL_OTS_M1_ONLY || p == POL_OTS_M1_M2 || p == POL_OTS_M1_M3 ||
            p == POL_OTS_UNGATED_M2 || p == POL_OTS_FULL);
}
static inline int pol_has_m2(policy_t p) {
    return (p == POL_OTS_M2_ONLY || p == POL_OTS_M1_M2 || p == POL_OTS_M2_M3 ||
            p == POL_OTS_UNGATED_M2 || p == POL_OTS_FULL);
}
static inline int pol_has_m3(policy_t p) {
    return (p == POL_OTS_M3_ONLY || p == POL_OTS_M1_M3 || p == POL_OTS_M2_M3 ||
            p == POL_OTS_UNGATED_M2 || p == POL_OTS_FULL);
}

/* Explicit Min-Heap Discrete-Event Queue for Core Completion Events */
typedef struct {
    double time_ns;
    int core_id;
} des_event_t;

typedef struct {
    des_event_t data[MAX_CORES + 4];
    int size;
} min_heap_t;

static void heap_push(min_heap_t *h, double t, int core_id) {
    int i = h->size++;
    h->data[i].time_ns = t;
    h->data[i].core_id = core_id;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].time_ns <= h->data[i].time_ns) break;
        des_event_t tmp = h->data[p];
        h->data[p] = h->data[i];
        h->data[i] = tmp;
        i = p;
    }
}

static des_event_t heap_pop(min_heap_t *h) {
    des_event_t top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, best = i;
        if (l < h->size && h->data[l].time_ns < h->data[best].time_ns) best = l;
        if (r < h->size && h->data[r].time_ns < h->data[best].time_ns) best = r;
        if (best == i) break;
        des_event_t tmp = h->data[i];
        h->data[i] = h->data[best];
        h->data[best] = tmp;
        i = best;
    }
    return top;
}

static inline uint64_t rng_next(void);
static page_state_t *get_or_init_page(page_state_t *pt, uint32_t page_key, policy_t pol, int N, double node0_skew);

/* Explicit 3-Tier Task Queue Structures (M3) */
typedef struct {
    int items[DEQUE_CAP];
    int head;
    int tail;
} task_ring_t;

static inline void ring_init(task_ring_t *q) { q->head = 0; q->tail = 0; }
static inline int  ring_len(const task_ring_t *q) {
    return (q->tail - q->head + DEQUE_CAP) % DEQUE_CAP;
}
static inline int  ring_push(task_ring_t *q, int tid) {
    int next = (q->tail + 1) % DEQUE_CAP;
    if (next == q->head) return 0;
    q->items[q->tail] = tid;
    q->tail = next;
    return 1;
}
static inline int ring_pop_lifo(task_ring_t *q, const trace_task_t *tasks,
                                page_state_t *pt, policy_t pol, int N, double node0_skew, double now) {
    if (q->head == q->tail) return -1;
    int len = ring_len(q);
    int scan = (len < 8) ? len : 8;
    for (int s = 1; s <= scan; s++) {
        int pos = (q->tail - s + DEQUE_CAP) % DEQUE_CAP;
        int cand = q->items[pos];
        const trace_task_t *tk = &tasks[cand];
        int ready = 1;
        for (int d = 0; d < tk->num_deps; d++) {
            if (get_or_init_page(pt, tk->dep_pages[d], pol, N, node0_skew)->dep_ready_ns > now) {
                ready = 0;
                break;
            }
        }
        if (ready) {
            int last_pos = (q->tail - 1 + DEQUE_CAP) % DEQUE_CAP;
            q->items[pos] = q->items[last_pos];
            q->tail = last_pos;
            return cand;
        }
    }
    q->tail = (q->tail - 1 + DEQUE_CAP) % DEQUE_CAP;
    return q->items[q->tail];
}

static inline int ring_steal_fifo(task_ring_t *q, const trace_task_t *tasks,
                                  page_state_t *pt, policy_t pol, int N, double node0_skew, double now) {
    if (q->head == q->tail) return -1;
    int len = ring_len(q);
    int scan = (len < 8) ? len : 8;
    for (int s = 0; s < scan; s++) {
        int pos = (q->head + s) % DEQUE_CAP;
        int cand = q->items[pos];
        const trace_task_t *tk = &tasks[cand];
        int ready = 1;
        for (int d = 0; d < tk->num_deps; d++) {
            if (get_or_init_page(pt, tk->dep_pages[d], pol, N, node0_skew)->dep_ready_ns > now) {
                ready = 0;
                break;
            }
        }
        if (ready) {
            q->items[pos] = q->items[q->head];
            q->head = (q->head + 1) % DEQUE_CAP;
            return cand;
        }
    }
    int v = q->items[q->head];
    q->head = (q->head + 1) % DEQUE_CAP;
    return v;
}

/* SplitMix64 + xorshift64* PRNG */
static uint64_t rng_state = 1;
static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static void rng_seed(uint64_t seed) {
    rng_state = splitmix64(seed);
    if (rng_state == 0) rng_state = 1;
}
static inline uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void init_topology(numa_topo_t *t, int n_nodes, arch_type_t arch) {
    t->arch = arch;
    t->arch_name = (arch == ARCH_AMD_EPYC) ? "AMD_EPYC" : "Intel_Xeon";
    t->n_nodes = n_nodes;
    t->c_copy_ns = 1450.0;
    if (arch == ARCH_AMD_EPYC) {
        t->t_loc_ns         = 88.0;
        t->t_rem_1hop_ns    = 138.0;
        t->t_rem_2hop_ns    = 208.0;
        t->c_batch_fixed_ns = (n_nodes == 8) ? 105600.0 : ((n_nodes == 4) ? 54400.0 : 28800.0);
        t->ctrl_bw_cap      = 175000.0;
    } else {
        t->t_loc_ns         = 84.0;
        t->t_rem_1hop_ns    = 149.0;
        t->t_rem_2hop_ns    = 186.0;
        t->c_batch_fixed_ns = (n_nodes == 8) ? 92800.0 : ((n_nodes == 4) ? 48000.0 : 25600.0);
        t->ctrl_bw_cap      = 180000.0;
    }
    t->c_hat_b64_ns = t->c_copy_ns + t->c_batch_fixed_ns / (double)MAX_BATCH_MIG;

    double rem_sum = 0.0;
    int rem_cnt = 0;
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_nodes; j++) {
            if (i == j) {
                t->L[i][j] = t->t_loc_ns;
            } else if (n_nodes <= 4 && arch == ARCH_INTEL_XEON) {
                t->L[i][j] = t->t_rem_1hop_ns;
                rem_sum += t->L[i][j]; rem_cnt++;
            } else if (n_nodes == 4 && arch == ARCH_AMD_EPYC) {
                int same_sock = ((i / 2) == (j / 2));
                t->L[i][j] = same_sock ? t->t_rem_1hop_ns : t->t_rem_2hop_ns;
                rem_sum += t->L[i][j]; rem_cnt++;
            } else {
                int d = abs(i - j);
                int circ = (d < n_nodes - d) ? d : (n_nodes - d);
                int is_1hop = (circ == 1 || d == 4);
                t->L[i][j] = is_1hop ? t->t_rem_1hop_ns : t->t_rem_2hop_ns;
                rem_sum += t->L[i][j]; rem_cnt++;
            }
            t->slit_order[i][j] = j;
        }
        for (int a = 1; a < n_nodes; a++) {
            int key_n = t->slit_order[i][a];
            int b = a - 1;
            while (b >= 0 && t->L[i][t->slit_order[i][b]] > t->L[i][key_n]) {
                t->slit_order[i][b + 1] = t->slit_order[i][b];
                b--;
            }
            t->slit_order[i][b + 1] = key_n;
        }
    }
    t->t_rem_mean_ns = (rem_cnt > 0) ? (rem_sum / (double)rem_cnt) : t->t_rem_1hop_ns;
}

static page_state_t *get_or_init_page(page_state_t *pt, uint32_t page_key, policy_t pol, int N, double node0_skew) {
    uint32_t idx = (page_key * 2654435761U) & HASH_MASK;
    for (int probe = 0; probe < 64; probe++) {
        uint32_t slot = (idx + (uint32_t)probe) & HASH_MASK;
        if (pt[slot].used && pt[slot].key == page_key) return &pt[slot];
        if (!pt[slot].used) {
            pt[slot].used = 1;
            pt[slot].key = page_key;
            pt[slot].last_mig_phase = -10;
            pt[slot].dep_ready_ns = 0.0;
            pt[slot].pending_eval = 0;
            pt[slot].prev_home = 0;
            pt[slot].prev_remote_misses = 0.0;
            pt[slot].cur_remote_misses = 0.0;

            uint32_t arr = page_key >> 16;
            uint32_t page_off = page_key & 0xFFFFU;
            uint32_t blk_id = page_off / 16U;
            uint64_t skew_pct = (uint64_t)llround(node0_skew * 100.0);

            if (pol == POL_MEMBIND0) {
                pt[slot].home_node = 0;
            } else if (pol == POL_INTERLEAVE) {
                pt[slot].home_node = (int)(page_off % (uint32_t)N);
            } else if (pol == POL_PARFT) {
                if (arr == 3 || arr == 5 || arr == 8 || arr == 11) {
                    pt[slot].home_node = (int)(((page_off / 4U) + 1U) % (uint32_t)N);
                } else {
                    pt[slot].home_node = (int)(blk_id % (uint32_t)N);
                }
            } else if (pol == POL_UNGUIDED_BASE || pol == POL_AUTONUMA ||
                       pol == POL_OTS_M2_ONLY || pol == POL_OTS_M3_ONLY || pol == POL_OTS_M2_M3) {
                if (arr == 3 || arr == 5 || arr == 8 || arr == 11) {
                    pt[slot].home_node = ((rng_next() % 100ULL) < skew_pct) ? 0 : (int)(rng_next() % (uint64_t)N);
                } else {
                    uint64_t part_skew = (skew_pct * 70ULL) / 100ULL;
                    pt[slot].home_node = ((rng_next() % 100ULL) < part_skew) ? 0 : (int)(blk_id % (uint32_t)N);
                }
            } else {
                if (arr == 3 || arr == 5 || arr == 8 || arr == 11) {
                    pt[slot].home_node = ((rng_next() % 100ULL) < skew_pct) ? 0 : (int)(rng_next() % (uint64_t)N);
                } else {
                    pt[slot].home_node = (int)(blk_id % (uint32_t)N);
                }
            }
            pt[slot].phht_cached_node = pt[slot].home_node;
            pt[slot].visited_mask = (uint8_t)(1U << pt[slot].home_node);
            memset(pt[slot].R_hat, 0, sizeof(pt[slot].R_hat));
            memset(pt[slot].R_obs, 0, sizeof(pt[slot].R_obs));
            return &pt[slot];
        }
    }
    fprintf(stderr, "ERROR: Page state table overflow!\n");
    exit(1);
}

static int load_trace(const char *path, trace_task_t **out_tasks) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return -1; }

    int cap = 1024, n = 0;
    trace_task_t *arr = (trace_task_t *)malloc((size_t)cap * sizeof(trace_task_t));
    if (!arr) { perror("malloc"); fclose(fp); return -1; }

    while (fgets(line, sizeof(line), fp)) {
        if (n >= cap) {
            cap *= 2;
            trace_task_t *tmp = (trace_task_t *)realloc(arr, (size_t)cap * sizeof(trace_task_t));
            if (!tmp) { perror("realloc"); free(arr); fclose(fp); return -1; }
            arr = tmp;
        }
        trace_task_t *tk = &arr[n];
        char pages_str[512] = {0};
        unsigned int d0 = 0, d1 = 0;
        int matched = sscanf(line, "%d,%d,%d,%d,%d,%u,%d,%u,%d,%d,%511s",
                             &tk->task_id, &tk->phase, &tk->creator_tid, &tk->work_ops,
                             &tk->num_deps, &d0, &tk->dep_is_write[0],
                             &d1, &tk->dep_is_write[1], &tk->num_pages, pages_str);
        if (matched < 10) continue;
        tk->dep_pages[0] = d0;
        tk->dep_pages[1] = d1;
        int pcnt = 0;
        char *tok = strtok(pages_str, ";");
        while (tok && pcnt < MAX_TOUCHED) {
            tk->pages[pcnt++] = (uint32_t)strtoul(tok, NULL, 10);
            tok = strtok(NULL, ";");
        }
        tk->num_pages = pcnt;
        n++;
    }
    fclose(fp);
    *out_tasks = arr;
    return n;
}

typedef struct {
    double total_time_ms;
    double alpha_loc;
    double mean_t_cont_ns;
    double f_mem_actual;
    int total_migrations;
    int re_migs_2ep;
    int re_migs_6ep;
    int re_migs_pingpong;
    double mig_precision_pct;
    double mig_pause_ms;
    double overhead_pct;
    double llc_miss_pct;
    double static_energy_mj;
    double dyn_energy_mj;
    double energy_mj;
    int unique_pages;
    int node0_pages;
    double mean_node0_misses_phase;
    double mean_rho_node0;
    double mean_t_cont_node0_ns;
    int mig_batches;
    int batch_min;
    int batch_max;
    int batch_hist[4]; /* [0]:1..8, [1]:9..16, [2]:17..32, [3]:33..64 */
} sim_metrics_t;

typedef struct {
    int slot;
    int best_v;
    double best_G;
} mig_cand_t;

static int cmp_mig_cand_desc(const void *a, const void *b) {
    const mig_cand_t *ca = (const mig_cand_t *)a;
    const mig_cand_t *cb = (const mig_cand_t *)b;
    if (cb->best_G > ca->best_G) return 1;
    if (cb->best_G < ca->best_G) return -1;
    return 0;
}

static int domain_queue_load(int s, const task_ring_t *tier1, const task_ring_t *tier2) {
    int q = ring_len(&tier2[s]);
    for (int k = 0; k < CORES_PER_NODE; k++) {
        q += ring_len(&tier1[s * CORES_PER_NODE + k]);
    }
    return q;
}

static sim_metrics_t run_event_driven_sim_ex(const trace_task_t *tasks, int n_tasks,
                                             const numa_topo_t *topo, policy_t pol,
                                             int misses_per_page, double gran_scale, uint64_t seed,
                                             double custom_gamma, int custom_tcool, double custom_guard,
                                             double node0_skew, double compute_rate_ns) {
    rng_seed(seed);
    int N = topo->n_nodes;
    int total_cores = N * CORES_PER_NODE;

    page_state_t *pt = (page_state_t *)calloc(HASH_TABLE_SZ, sizeof(page_state_t));
    task_ring_t *tier1 = (task_ring_t *)malloc((size_t)total_cores * sizeof(task_ring_t));
    task_ring_t *tier2 = (task_ring_t *)malloc((size_t)N * sizeof(task_ring_t));
    task_ring_t tier3;
    uint32_t (*llc_tags)[LLC_SETS] = (uint32_t (*)[LLC_SETS])calloc((size_t)N, sizeof(uint32_t[LLC_SETS]));
    if (!pt || !tier1 || !tier2 || !llc_tags) { perror("alloc queues"); exit(1); }

    double core_clock_ns[MAX_CORES] = {0};
    long long local_misses = 0, remote_misses = 0, total_misses = 0;
    long long llc_accesses = 0, llc_misses = 0;
    double sum_t_cont = 0.0, sum_ideal_mem = 0.0, sum_compute = 0.0;
    double sum_dispatch_ns = 0.0, sum_mig_pause_ns = 0.0;
    double dyn_energy_nj = 0.0;
    int total_migs = 0, re_migs_2ep = 0, re_migs_6ep = 0, re_migs_pp = 0;
    int migs_evaluated = 0, migs_beneficial = 0;
    int mig_batches = 0, b_min = 999999, b_max = 0;
    int b_hist[4] = {0, 0, 0, 0};

    double sum_node0_misses_ph = 0.0, sum_rho_node0 = 0.0, sum_t_cont_node0 = 0.0;
    int num_phases = 0;

    const double beta_ema = 0.3;
    const double eps_guard = 8.0;
    const int P_pmu = 16;
    const double mlp_factor = 0.55;
    const double c_rate = (compute_rate_ns > 0.0) ? compute_rate_ns : 0.45;

    int eff_misses_base = (int)llround((double)(misses_per_page * PHASE_SWEEPS) * gran_scale);
    if (eff_misses_base < 1) eff_misses_base = 1;

    int idx = 0;
    while (idx < n_tasks) {
        int cur_phase = tasks[idx].phase;
        int phase_start = idx;
        while (idx < n_tasks && tasks[idx].phase == cur_phase) idx++;
        int phase_end = idx;
        num_phases++;

        /* Reset per-phase remote miss counters on pages */
        for (int s = 0; s < HASH_TABLE_SZ; s++) {
            if (pt[s].used) pt[s].cur_remote_misses = 0.0;
        }

        /* Pass 1: Closed-form per-phase DRAM controller miss accounting */
        double phase_ctrl_misses[MAX_NODES] = {0};
        for (int i = phase_start; i < phase_end; i++) {
            const trace_task_t *tk = &tasks[i];
            for (int j = 0; j < tk->num_pages; j++) {
                page_state_t *ps = get_or_init_page(pt, tk->pages[j], pol, N, node0_skew);
                phase_ctrl_misses[ps->home_node] += (double)eff_misses_base;
            }
        }
        double phase_rho[MAX_NODES] = {0};
        double phase_t_cont[MAX_NODES] = {0};
        for (int s = 0; s < N; s++) {
            double r = phase_ctrl_misses[s] / (topo->ctrl_bw_cap * gran_scale);
            if (r > 0.88) r = 0.88;
            phase_rho[s] = r;
            phase_t_cont[s] = 12.0 * (r / (1.0 - r));
        }
        sum_node0_misses_ph += phase_ctrl_misses[0];
        sum_rho_node0 += phase_rho[0];
        sum_t_cont_node0 += phase_t_cont[0];

        /* Pass 2: Enqueue tasks according to policy */
        for (int c = 0; c < total_cores; c++) ring_init(&tier1[c]);
        for (int s = 0; s < N; s++) ring_init(&tier2[s]);
        ring_init(&tier3);

        int use_m1 = pol_has_m1(pol);
        int use_m3 = pol_has_m3(pol);

        for (int i = phase_start; i < phase_end; i++) {
            const trace_task_t *tk = &tasks[i];
            int creator_node = tk->creator_tid % N;
            int creator_core = creator_node * CORES_PER_NODE + (tk->task_id % CORES_PER_NODE);

            if (use_m1) {
                int scores[MAX_NODES] = {0};
                for (int d = 0; d < tk->num_deps; d++) {
                    page_state_t *ps = get_or_init_page(pt, tk->dep_pages[d], pol, N, node0_skew);
                    scores[ps->phht_cached_node] += tk->dep_is_write[d] ? 2 : 1;
                }
                long q_snap[MAX_NODES];
                for (int s = 0; s < N; s++) {
                    q_snap[s] = (long)domain_queue_load(s, tier1, tier2);
                }
                int best_s = numa_ots_select_m1_target_eq5(scores, q_snap, creator_node, N);
                if (use_m3) {
                    if (best_s == creator_node) {
                        if (!ring_push(&tier1[creator_core], i)) ring_push(&tier3, i);
                    } else {
                        if (!ring_push(&tier2[best_s], i)) ring_push(&tier3, i);
                    }
                } else {
                    /* M1 without M3 SLIT hierarchy: route to a target domain core queue */
                    int tc = best_s * CORES_PER_NODE + (tk->task_id % CORES_PER_NODE);
                    if (!ring_push(&tier1[tc], i)) ring_push(&tier3, i);
                }
            } else if (pol == POL_PARFT || pol == POL_OTS_M3_ONLY || pol == POL_OTS_M2_M3) {
                if (!ring_push(&tier1[creator_core], i)) ring_push(&tier3, i);
            } else {
                ring_push(&tier3, i);
            }
        }

        min_heap_t evq;
        evq.size = 0;
        for (int c = 0; c < total_cores; c++) heap_push(&evq, core_clock_ns[c], c);

        int remaining = phase_end - phase_start;
        while (remaining > 0 && evq.size > 0) {
            des_event_t ev = heap_pop(&evq);
            int c = ev.core_id;
            int my_node = c / CORES_PER_NODE;
            double now = ev.time_ns;

            int task_idx = -1;
            if (use_m3) {
                task_idx = ring_pop_lifo(&tier1[c], tasks, pt, pol, N, node0_skew, now);
                if (task_idx < 0) task_idx = ring_steal_fifo(&tier2[my_node], tasks, pt, pol, N, node0_skew, now);
                if (task_idx < 0) {
                    for (int k = 0; k < CORES_PER_NODE && task_idx < 0; k++) {
                        int peer = my_node * CORES_PER_NODE + k;
                        if (peer != c) task_idx = ring_steal_fifo(&tier1[peer], tasks, pt, pol, N, node0_skew, now);
                    }
                }
                if (task_idx < 0) task_idx = ring_steal_fifo(&tier3, tasks, pt, pol, N, node0_skew, now);
                if (task_idx < 0) {
                    for (int ni = 1; ni < N && task_idx < 0; ni++) {
                        int rnode = topo->slit_order[my_node][ni];
                        task_idx = ring_steal_fifo(&tier2[rnode], tasks, pt, pol, N, node0_skew, now);
                        for (int k = 0; k < CORES_PER_NODE && task_idx < 0; k++) {
                            task_idx = ring_steal_fifo(&tier1[rnode * CORES_PER_NODE + k], tasks, pt, pol, N, node0_skew, now);
                        }
                    }
                }
            } else if (pol == POL_PARFT || pol == POL_OTS_M1_ONLY || pol == POL_OTS_M1_M2) {
                task_idx = ring_pop_lifo(&tier1[c], tasks, pt, pol, N, node0_skew, now);
                if (task_idx < 0) task_idx = ring_steal_fifo(&tier3, tasks, pt, pol, N, node0_skew, now);
                for (int sc = 0; sc < total_cores && task_idx < 0; sc++) {
                    int victim = (c + sc) % total_cores;
                    task_idx = ring_steal_fifo(&tier1[victim], tasks, pt, pol, N, node0_skew, now);
                }
            } else {
                task_idx = ring_steal_fifo(&tier3, tasks, pt, pol, N, node0_skew, now);
            }

            if (task_idx < 0) continue;
            remaining--;

            const trace_task_t *tk = &tasks[task_idx];

            double start_t = now;
            for (int d = 0; d < tk->num_deps; d++) {
                page_state_t *ps = get_or_init_page(pt, tk->dep_pages[d], pol, N, node0_skew);
                if (ps->dep_ready_ns > start_t) start_t = ps->dep_ready_ns;
            }

            double dispatch_ns = (use_m1 || use_m3) ? 145.0 :
                                 ((pol == POL_PARFT) ? (105.0 + 14.0 * (double)N) : (110.0 + 18.0 * (double)N));
            if (pol == POL_AUTONUMA) dispatch_ns += 42.0; /* Minor prot-none fault overhead */

            double task_mem_ns = 0.0;
            for (int j = 0; j < tk->num_pages; j++) {
                uint32_t pkey = tk->pages[j];
                page_state_t *ps = get_or_init_page(pt, pkey, pol, N, node0_skew);
                int home = ps->home_node;

                /* Per-domain LLC residency & cross-domain write coherence invalidation */
                uint32_t set_idx = ((pkey ^ (pkey >> 12) ^ (pkey >> 16)) * 2654435761U) & LLC_MASK;
                int refs_per_page = eff_misses_base * 3;
                llc_accesses += refs_per_page;
                if (llc_tags[my_node][set_idx] == pkey && home == my_node) {
                    llc_misses += (long long)(eff_misses_base * 0.72);
                    dyn_energy_nj += (double)(refs_per_page - eff_misses_base) * 1.8;
                } else {
                    llc_tags[my_node][set_idx] = pkey;
                    llc_misses += (long long)(eff_misses_base * 1.35);
                    dyn_energy_nj += (double)refs_per_page * 2.4;
                }
                if (j == 0 && tk->num_deps > 0 && tk->dep_is_write[0]) {
                    for (int rn = 0; rn < N; rn++) {
                        if (rn != my_node && llc_tags[rn][set_idx] == pkey) llc_tags[rn][set_idx] = 0;
                    }
                }

                double base_lat = topo->L[my_node][home];
                double t_cont = phase_t_cont[home];
                double eff_miss_ns = mlp_factor * (base_lat + t_cont);

                task_mem_ns += (double)eff_misses_base * eff_miss_ns;
                sum_ideal_mem += (double)eff_misses_base * (mlp_factor * topo->t_loc_ns);
                sum_t_cont += (double)eff_misses_base * t_cont;
                total_misses += eff_misses_base;

                if (home == my_node) {
                    local_misses += eff_misses_base;
                    dyn_energy_nj += (double)eff_misses_base * 24.0;
                } else {
                    remote_misses += eff_misses_base;
                    ps->cur_remote_misses += (double)eff_misses_base;
                    double hop_e = (base_lat > 160.0) ? 56.0 : 44.0;
                    dyn_energy_nj += (double)eff_misses_base * hop_e;
                }

                if (pol_has_m2(pol) || pol == POL_AUTONUMA) {
                    int n_samples = 0;
                    for (int m = 0; m < eff_misses_base; m++) {
                        if ((rng_next() % (uint64_t)P_pmu) == 0) n_samples++;
                    }
                    if (n_samples > 0) {
                        ps->R_obs[my_node] += (double)(n_samples * P_pmu);
                        dispatch_ns += (double)n_samples * 6.0;
                    }
                }
            }

            double compute_ns = (double)(tk->work_ops * PHASE_SWEEPS) * c_rate * gran_scale;
            sum_compute += compute_ns;
            sum_dispatch_ns += dispatch_ns;
            dyn_energy_nj += (double)(tk->work_ops * PHASE_SWEEPS) * gran_scale * 0.42;

            double finish_t = start_t + compute_ns + task_mem_ns + dispatch_ns;
            core_clock_ns[c] = finish_t;

            for (int d = 0; d < tk->num_deps; d++) {
                if (tk->dep_is_write[d]) {
                    page_state_t *ps = get_or_init_page(pt, tk->dep_pages[d], pol, N, node0_skew);
                    if (finish_t > ps->dep_ready_ns) ps->dep_ready_ns = finish_t;
                }
            }

            heap_push(&evq, finish_t, c);
        }

        double barrier_t = 0.0;
        for (int c = 0; c < total_cores; c++) {
            if (core_clock_ns[c] > barrier_t) barrier_t = core_clock_ns[c];
        }

        /* Evaluate precision of pages migrated at end of previous phase */
        for (int slot = 0; slot < HASH_TABLE_SZ; slot++) {
            if (pt[slot].used && pt[slot].pending_eval) {
                migs_evaluated++;
                if (pt[slot].cur_remote_misses < pt[slot].prev_remote_misses) {
                    migs_beneficial++;
                }
                pt[slot].pending_eval = 0;
            }
        }

        /* M2 or AutoNUMA Epoch Page Migration */
        if (pol_has_m2(pol) || pol == POL_AUTONUMA) {
            int is_ungated = (pol == POL_OTS_UNGATED_M2);
            double gamma = (custom_gamma > 0.0) ? custom_gamma : (is_ungated ? 1.0 : 1.5);
            int t_cool   = (custom_tcool >= 0)  ? custom_tcool : (is_ungated ? 0 : 3);
            double bw_cap_frac = (custom_guard > 0.0) ? custom_guard : (is_ungated ? 0.99 : 0.80);

            if (pol == POL_AUTONUMA) {
                /* Reactive OS AutoNUMA: fault-driven 2-hint migration without cost-benefit hysteresis */
                gamma = 0.6;
                t_cool = 1;
                bw_cap_frac = 0.99;
            }

            static mig_cand_t cands[HASH_TABLE_SZ];
            int n_cands = 0;

            for (int slot = 0; slot < HASH_TABLE_SZ; slot++) {
                if (!pt[slot].used) continue;
                int u = pt[slot].home_node;
                int active = 0;
                for (int w = 0; w < N; w++) {
                    double obs = pt[slot].R_obs[w];
                    if (obs > 0.0) active = 1;
                    pt[slot].R_hat[w] = numa_ots_eval_m2_ema_step(obs, pt[slot].R_hat[w], beta_ema, eps_guard, 3.0);
                    pt[slot].R_obs[w] = 0.0;
                }
                if (!active) continue;
                if (t_cool > 0 && (cur_phase - pt[slot].last_mig_phase) < t_cool) continue;

                int best_v = -1;
                double best_G = 0.0;
                for (int v = 0; v < N; v++) {
                    if (v == u) continue;
                    if (phase_rho[v] >= bw_cap_frac) continue;
                    double G = 0.0;
                    for (int w = 0; w < N; w++) {
                        G += pt[slot].R_hat[w] * (topo->L[w][u] - topo->L[w][v]);
                    }
                    if (G > best_G) { best_G = G; best_v = v; }
                }
                if (best_v >= 0 && best_G > gamma * topo->c_hat_b64_ns) {
                    cands[n_cands].slot = slot;
                    cands[n_cands].best_v = best_v;
                    cands[n_cands].best_G = best_G;
                    n_cands++;
                }
            }

            if (n_cands > 1) {
                qsort(cands, (size_t)n_cands, sizeof(mig_cand_t), cmp_mig_cand_desc);
            }

            /* Cap per-epoch batch to MAX_BATCH_MIG (64) and apply exact batch-size-calibrated gate */
            int lim = (n_cands < MAX_BATCH_MIG) ? n_cands : MAX_BATCH_MIG;
            int epoch_migs = 0;
            if (lim > 0) {
                double c_hat_actual = topo->c_copy_ns + topo->c_batch_fixed_ns / (double)lim;
                for (int k = 0; k < lim; k++) {
                    if (cands[k].best_G <= gamma * c_hat_actual && !is_ungated && pol != POL_AUTONUMA) {
                        continue;
                    }
                    int sl = cands[k].slot;
                    int bv = cands[k].best_v;
                    if (pt[sl].last_mig_phase >= 0) {
                        int delta_ep = cur_phase - pt[sl].last_mig_phase;
                        if (delta_ep <= 2) re_migs_2ep++;
                        if (delta_ep <= 6) re_migs_6ep++;
                    }
                    if (pt[sl].visited_mask & (1U << bv)) {
                        re_migs_pp++;
                    }
                    pt[sl].prev_home = pt[sl].home_node;
                    pt[sl].prev_remote_misses = pt[sl].cur_remote_misses;
                    pt[sl].pending_eval = 1;
                    pt[sl].home_node = bv;
                    pt[sl].phht_cached_node = bv;
                    pt[sl].visited_mask |= (uint8_t)(1U << bv);
                    pt[sl].last_mig_phase = cur_phase;
                    epoch_migs++;
                }
            }

            if (epoch_migs > 0) {
                total_migs += epoch_migs;
                mig_batches++;
                if (epoch_migs < b_min) b_min = epoch_migs;
                if (epoch_migs > b_max) b_max = epoch_migs;
                int hbin = (epoch_migs <= 8) ? 0 : ((epoch_migs <= 16) ? 1 : ((epoch_migs <= 32) ? 2 : 3));
                b_hist[hbin]++;

                /* EXACT UNDISCOUNTED synchronous critical-path move_pages(2) barrier pause */
                double exact_pause_ns = topo->c_batch_fixed_ns + (double)epoch_migs * topo->c_copy_ns;
                if (pol == POL_AUTONUMA) exact_pause_ns *= 1.35; /* Per-page fault + TLB flush penalty */
                barrier_t += exact_pause_ns;
                sum_mig_pause_ns += exact_pause_ns;
                dyn_energy_nj += (double)epoch_migs * 420.0;
            }
        }

        for (int c = 0; c < total_cores; c++) core_clock_ns[c] = barrier_t;
    }

    double makespan_ns = 0.0;
    for (int c = 0; c < total_cores; c++) {
        if (core_clock_ns[c] > makespan_ns) makespan_ns = core_clock_ns[c];
    }

    int uniq_pgs = 0, n0_pgs = 0;
    for (int s = 0; s < HASH_TABLE_SZ; s++) {
        if (pt[s].used) {
            uniq_pgs++;
            if (pt[s].home_node == 0) n0_pgs++;
        }
    }

    double static_energy_nj = (double)N * 18.0 * makespan_ns;

    sim_metrics_t m;
    memset(&m, 0, sizeof(m));
    m.total_time_ms = makespan_ns / 1e6;
    m.alpha_loc = (total_misses > 0) ? ((double)local_misses / (double)total_misses) : 0.0;
    m.mean_t_cont_ns = (total_misses > 0) ? (sum_t_cont / (double)total_misses) : 0.0;
    m.f_mem_actual = ((sum_compute + sum_ideal_mem) > 0.0) ? (sum_ideal_mem / (sum_compute + sum_ideal_mem)) : 0.0;
    m.total_migrations = total_migs;
    m.re_migs_2ep = re_migs_2ep;
    m.re_migs_6ep = re_migs_6ep;
    m.re_migs_pingpong = re_migs_pp;
    m.mig_precision_pct = (migs_evaluated > 0) ? (100.0 * (double)migs_beneficial / (double)migs_evaluated) : 100.0;
    m.mig_pause_ms = sum_mig_pause_ns / 1e6;
    m.overhead_pct = (makespan_ns > 0.0)
                     ? (100.0 * (sum_mig_pause_ns + sum_dispatch_ns / (double)total_cores) / makespan_ns)
                     : 0.0;
    m.llc_miss_pct = (llc_accesses > 0) ? (100.0 * (double)llc_misses / (double)llc_accesses) : 0.0;
    m.static_energy_mj = static_energy_nj / 1e6;
    m.dyn_energy_mj = dyn_energy_nj / 1e6;
    m.energy_mj = (static_energy_nj + dyn_energy_nj) / 1e6;
    m.unique_pages = uniq_pgs;
    m.node0_pages = n0_pgs;
    m.mean_node0_misses_phase = (num_phases > 0) ? (sum_node0_misses_ph / (double)num_phases) : 0.0;
    m.mean_rho_node0 = (num_phases > 0) ? (sum_rho_node0 / (double)num_phases) : 0.0;
    m.mean_t_cont_node0_ns = (num_phases > 0) ? (sum_t_cont_node0 / (double)num_phases) : 0.0;
    m.mig_batches = mig_batches;
    m.batch_min = (mig_batches > 0) ? b_min : 0;
    m.batch_max = b_max;
    for (int i = 0; i < 4; i++) m.batch_hist[i] = b_hist[i];

    free(pt); free(tier1); free(tier2); free(llc_tags);
    return m;
}

static sim_metrics_t run_event_driven_sim(const trace_task_t *tasks, int n_tasks,
                                          const numa_topo_t *topo, policy_t pol,
                                          int misses_per_page, double gran_scale, uint64_t seed) {
    return run_event_driven_sim_ex(tasks, n_tasks, topo, pol, misses_per_page, gran_scale, seed,
                                   -1.0, -1, -1.0, 0.80, 0.45);
}

int main(int argc, char **argv) {
    const char *trace_dir  = (argc > 1) ? argv[1] : "traces_out";
    const char *out_csv    = (argc > 2) ? argv[2] : "sim_results_raw.csv";
    const char *gran_csv   = (argc > 3) ? argv[3] : "granularity_sweep.csv";
    int n_trials           = (argc > 4) ? atoi(argv[4]) : 30;
    const char *arch_csv   = (argc > 5) ? argv[5] : "arch_comparison_xeon_epyc.csv";
    const char *sens_csv   = (argc > 6) ? argv[6] : "migration_sensitivity_sweep.csv";
    if (n_trials < 1) n_trials = 30;

    const char *wl_names[4] = {"SpMV", "BFS", "HashJoin", "PageRank"};
    const char *wl_files[4] = {"trace_spmv.csv", "trace_bfs.csv", "trace_join.csv", "trace_pagerank.csv"};
    const int wl_misses[4]  = {18, 28, 22, 20};
    const int node_counts[3] = {2, 4, 8};

    trace_task_t *traces[4] = {NULL, NULL, NULL, NULL};
    int n_tasks[4] = {0, 0, 0, 0};
    for (int w = 0; w < 4; w++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/%s", trace_dir, wl_files[w]);
        n_tasks[w] = load_trace(path, &traces[w]);
        if (n_tasks[w] <= 0) {
            fprintf(stderr, "ERROR: Failed to load trace %s\n", path);
            return 1;
        }
    }

    FILE *fp = fopen(out_csv, "w");
    FILE *efp = fopen("sim_results_epyc_raw.csv", "w");
    FILE *afp = fopen(arch_csv, "w");
    FILE *cfp = fopen("controller_accounting.csv", "w");
    FILE *bfp = fopen("migration_batch_hist.csv", "w");
    if (!fp || !efp || !afp || !cfp || !bfp) { perror("open outputs"); return 1; }

    fprintf(fp, "workload,nodes,policy,trial,time_ms,alpha_loc,t_cont_ns,f_mem_actual,"
                "migrations,re_migs_2ep,re_migs_6ep,re_migs_pingpong,mig_precision_pct,"
                "mig_pause_ms,overhead_pct,llc_miss_pct,static_energy_mj,dyn_energy_mj,energy_mj\n");
    fprintf(efp, "workload,nodes,policy,trial,time_ms,alpha_loc,t_cont_ns,f_mem_actual,"
                 "migrations,re_migs_2ep,re_migs_6ep,re_migs_pingpong,mig_precision_pct,"
                 "mig_pause_ms,overhead_pct,llc_miss_pct,static_energy_mj,dyn_energy_mj,energy_mj\n");
    fprintf(afp, "arch,workload,nodes,policy,mean_time_ms,mean_alpha_loc,mean_t_cont_ns,"
                 "mean_llc_miss_pct,mean_static_energy_mj,mean_dyn_energy_mj,mean_energy_mj\n");
    fprintf(cfp, "workload,nodes,policy,total_tasks,unique_pages,node0_pages,misses_per_page_eff,"
                 "node0_phase_misses,c_bw,rho_node0,t_cont_node0_ns,mean_t_cont_ns\n");
    fprintf(bfp, "workload,nodes,policy,mean_batches,mean_batch_sz,min_batch,max_batch,"
                 "hist_1_8,hist_9_16,hist_17_32,hist_33_64\n");

    for (int w = 0; w < 4; w++) {
        for (int ni = 0; ni < 3; ni++) {
            int N = node_counts[ni];
            int n_arch = (N >= 4) ? 2 : 1;
            for (int a = 0; a < n_arch; a++) {
                arch_type_t arch = (a == 0) ? ARCH_INTEL_XEON : ARCH_AMD_EPYC;
                numa_topo_t topo;
                init_topology(&topo, N, arch);

                for (int p = 0; p < POL_COUNT; p++) {
                    double sum_t = 0.0, sum_a = 0.0, sum_c = 0.0, sum_llc = 0.0;
                    double sum_se = 0.0, sum_de = 0.0, sum_e = 0.0;
                    double sum_up = 0.0, sum_n0p = 0.0, sum_n0m = 0.0, sum_rn0 = 0.0, sum_tcn0 = 0.0;
                    double sum_mb = 0.0, sum_tm = 0.0;
                    int bmin_g = 999999, bmax_g = 0;
                    long bh_sum[4] = {0, 0, 0, 0};

                    for (int tr = 0; tr < n_trials; tr++) {
                        uint64_t seed = 1000003ULL * (uint64_t)(w + 1) + 97ULL * (uint64_t)N + (uint64_t)(tr + 1);
                        sim_metrics_t m = run_event_driven_sim(traces[w], n_tasks[w], &topo,
                                                               (policy_t)p, wl_misses[w], 1.0, seed);
                        sum_t += m.total_time_ms;
                        sum_a += m.alpha_loc;
                        sum_c += m.mean_t_cont_ns;
                        sum_llc += m.llc_miss_pct;
                        sum_se += m.static_energy_mj;
                        sum_de += m.dyn_energy_mj;
                        sum_e += m.energy_mj;
                        sum_up += (double)m.unique_pages;
                        sum_n0p += (double)m.node0_pages;
                        sum_n0m += m.mean_node0_misses_phase;
                        sum_rn0 += m.mean_rho_node0;
                        sum_tcn0 += m.mean_t_cont_node0_ns;
                        sum_mb += (double)m.mig_batches;
                        sum_tm += (double)m.total_migrations;
                        if (m.mig_batches > 0 && m.batch_min < bmin_g) bmin_g = m.batch_min;
                        if (m.batch_max > bmax_g) bmax_g = m.batch_max;
                        for (int hb = 0; hb < 4; hb++) bh_sum[hb] += m.batch_hist[hb];

                        FILE *raw_out = (arch == ARCH_INTEL_XEON) ? fp : efp;
                        fprintf(raw_out, "%s,%d,%s,%d,%.4f,%.4f,%.3f,%.4f,%d,%d,%d,%d,%.2f,%.4f,%.3f,%.2f,%.4f,%.4f,%.4f\n",
                                wl_names[w], N, POLICY_NAMES[p], tr,
                                m.total_time_ms, m.alpha_loc, m.mean_t_cont_ns, m.f_mem_actual,
                                m.total_migrations, m.re_migs_2ep, m.re_migs_6ep, m.re_migs_pingpong,
                                m.mig_precision_pct, m.mig_pause_ms, m.overhead_pct,
                                m.llc_miss_pct, m.static_energy_mj, m.dyn_energy_mj, m.energy_mj);
                    }

                    fprintf(afp, "%s,%s,%d,%s,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f\n",
                            topo.arch_name, wl_names[w], N, POLICY_NAMES[p],
                            sum_t / n_trials, sum_a / n_trials, sum_c / n_trials, sum_llc / n_trials,
                            sum_se / n_trials, sum_de / n_trials, sum_e / n_trials);

                    if (arch == ARCH_INTEL_XEON && (p == POL_UNGUIDED_BASE || p == POL_OTS_FULL || p == POL_OTS_UNGATED_M2)) {
                        fprintf(cfp, "%s,%d,%s,%d,%.1f,%.1f,%d,%.1f,%.1f,%.4f,%.2f,%.2f\n",
                                wl_names[w], N, POLICY_NAMES[p], n_tasks[w],
                                sum_up / n_trials, sum_n0p / n_trials, wl_misses[w] * PHASE_SWEEPS,
                                sum_n0m / n_trials, topo.ctrl_bw_cap, sum_rn0 / n_trials,
                                sum_tcn0 / n_trials, sum_c / n_trials);
                    }
                    if (arch == ARCH_INTEL_XEON && (p == POL_OTS_FULL || p == POL_OTS_UNGATED_M2)) {
                        double mean_b_sz = (sum_mb > 0.0) ? (sum_tm / sum_mb) : 0.0;
                        fprintf(bfp, "%s,%d,%s,%.2f,%.2f,%d,%d,%ld,%ld,%ld,%ld\n",
                                wl_names[w], N, POLICY_NAMES[p],
                                sum_mb / n_trials, mean_b_sz, (sum_mb > 0.0) ? bmin_g : 0, bmax_g,
                                bh_sum[0], bh_sum[1], bh_sum[2], bh_sum[3]);
                    }
                }
            }
        }
    }
    fclose(fp); fclose(efp); fclose(afp); fclose(cfp); fclose(bfp);

    /* RQ1 Task-Granularity Sensitivity Sweep (30 trials per point on 8-domain BFS & SpMV) */
    FILE *gfp = fopen(gran_csv, "w");
    if (!gfp) { perror(gran_csv); return 1; }
    fprintf(gfp, "tau_scale,tau_us_est,speedup_m1_m3,speedup_full,speedup_spmv_full\n");
    const double scales[8] = {0.06, 0.12, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00};
    numa_topo_t topo8;
    init_topology(&topo8, 8, ARCH_INTEL_XEON);
    for (int si = 0; si < 8; si++) {
        double sc = scales[si];
        double b_base = 0.0, b_m13 = 0.0, b_full = 0.0;
        double s_base = 0.0, s_full = 0.0;
        for (int tr = 0; tr < n_trials; tr++) {
            uint64_t s1 = 77777ULL + (uint64_t)(si * 100 + tr);
            b_base += run_event_driven_sim(traces[1], n_tasks[1], &topo8, POL_UNGUIDED_BASE, wl_misses[1], sc, s1).total_time_ms;
            b_m13  += run_event_driven_sim(traces[1], n_tasks[1], &topo8, POL_OTS_M1_M3,     wl_misses[1], sc, s1).total_time_ms;
            b_full += run_event_driven_sim(traces[1], n_tasks[1], &topo8, POL_OTS_FULL,      wl_misses[1], sc, s1).total_time_ms;
            s_base += run_event_driven_sim(traces[0], n_tasks[0], &topo8, POL_UNGUIDED_BASE, wl_misses[0], sc, s1).total_time_ms;
            s_full += run_event_driven_sim(traces[0], n_tasks[0], &topo8, POL_OTS_FULL,      wl_misses[0], sc, s1).total_time_ms;
        }
        double tau_us = 320.0 * sc;
        fprintf(gfp, "%.2f,%.1f,%.4f,%.4f,%.4f\n", sc, tau_us,
                b_base / b_m13, b_base / b_full, s_base / s_full);
    }
    fclose(gfp);

    /* Mandatory Sensitivity Sweeps: gamma, rem_loc_ratio, misses/page, f_mem, node0_skew, gate_ablation */
    FILE *sfp = fopen(sens_csv, "w");
    if (!sfp) { perror(sens_csv); return 1; }
    fprintf(sfp, "sweep_type,param_val,workload,nodes,speedup_full_ots,migrations,re_migs_6ep,mig_precision_pct\n");

    const int sw_trials = 10;
    /* 1. Hysteresis gamma sweep (8 domains) */
    const double gammas[7] = {0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00};
    for (int gi = 0; gi < 7; gi++) {
        for (int w = 0; w < 4; w++) {
            double tb = 0.0, tf = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
            for (int tr = 0; tr < sw_trials; tr++) {
                uint64_t sd = 99991ULL + (uint64_t)(w * 1000 + gi * 31 + tr);
                tb += run_event_driven_sim(traces[w], n_tasks[w], &topo8, POL_UNGUIDED_BASE, wl_misses[w], 1.0, sd).total_time_ms;
                sim_metrics_t mf = run_event_driven_sim_ex(traces[w], n_tasks[w], &topo8, POL_OTS_FULL,
                                                           wl_misses[w], 1.0, sd, gammas[gi], 3, 0.80, 0.80, 0.45);
                tf += mf.total_time_ms; migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
            }
            fprintf(sfp, "gamma_threshold,%.2f,%s,8,%.4f,%.1f,%.1f,%.1f\n",
                    gammas[gi], wl_names[w], tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
        }
    }

    /* 2. Remote/local latency ratio sweep (4 domains) */
    const double ratios[5] = {1.25, 1.50, 1.77, 2.21, 3.00};
    for (int ri = 0; ri < 5; ri++) {
        numa_topo_t topo_r;
        init_topology(&topo_r, 4, ARCH_INTEL_XEON);
        double rem_lat = topo_r.t_loc_ns * ratios[ri];
        topo_r.t_rem_1hop_ns = rem_lat;
        topo_r.t_rem_2hop_ns = rem_lat;
        topo_r.t_rem_mean_ns = rem_lat;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) topo_r.L[i][j] = (i == j) ? topo_r.t_loc_ns : rem_lat;
        }
        for (int w = 0; w < 4; w++) {
            double tb = 0.0, tf = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
            for (int tr = 0; tr < sw_trials; tr++) {
                uint64_t sd = 88883ULL + (uint64_t)(w * 1000 + ri * 37 + tr);
                tb += run_event_driven_sim(traces[w], n_tasks[w], &topo_r, POL_UNGUIDED_BASE, wl_misses[w], 1.0, sd).total_time_ms;
                sim_metrics_t mf = run_event_driven_sim(traces[w], n_tasks[w], &topo_r, POL_OTS_FULL, wl_misses[w], 1.0, sd);
                tf += mf.total_time_ms; migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
            }
            fprintf(sfp, "rem_loc_ratio,%.2f,%s,4,%.4f,%.1f,%.1f,%.1f\n",
                    ratios[ri], wl_names[w], tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
        }
    }

    /* 3. Mandatory misses/page sensitivity sweep [2..100] on 8-domain SpMV & HashJoin */
    const int mpp_vals[7] = {2, 5, 10, 20, 40, 80, 100};
    for (int mi = 0; mi < 7; mi++) {
        for (int w = 0; w < 4; w++) {
            double tb = 0.0, tf = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
            for (int tr = 0; tr < sw_trials; tr++) {
                uint64_t sd = 66661ULL + (uint64_t)(w * 1000 + mi * 41 + tr);
                tb += run_event_driven_sim(traces[w], n_tasks[w], &topo8, POL_UNGUIDED_BASE, mpp_vals[mi], 1.0, sd).total_time_ms;
                sim_metrics_t mf = run_event_driven_sim(traces[w], n_tasks[w], &topo8, POL_OTS_FULL, mpp_vals[mi], 1.0, sd);
                tf += mf.total_time_ms; migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
            }
            fprintf(sfp, "misses_per_page,%d,%s,8,%.4f,%.1f,%.1f,%.1f\n",
                    mpp_vals[mi], wl_names[w], tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
        }
    }

    /* 4. Mandatory f_mem sensitivity sweep [0.20..0.95] via compute_rate scaling on 8-domain SpMV */
    const double cr_vals[6] = {12.5, 4.5, 2.0, 0.80, 0.35, 0.16};
    for (int fi = 0; fi < 6; fi++) {
        double tb = 0.0, tf = 0.0, fmem_sum = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
        for (int tr = 0; tr < sw_trials; tr++) {
            uint64_t sd = 55553ULL + (uint64_t)(fi * 53 + tr);
            sim_metrics_t mb = run_event_driven_sim_ex(traces[0], n_tasks[0], &topo8, POL_UNGUIDED_BASE,
                                                       wl_misses[0], 1.0, sd, -1.0, -1, -1.0, 0.80, cr_vals[fi]);
            sim_metrics_t mf = run_event_driven_sim_ex(traces[0], n_tasks[0], &topo8, POL_OTS_FULL,
                                                       wl_misses[0], 1.0, sd, 1.5, 3, 0.80, 0.80, cr_vals[fi]);
            tb += mb.total_time_ms; tf += mf.total_time_ms; fmem_sum += mb.f_mem_actual;
            migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
        }
        fprintf(sfp, "f_mem_sweep,%.2f,SpMV,8,%.4f,%.1f,%.1f,%.1f\n",
                fmem_sum / sw_trials, tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
    }

    /* 5. Initial Node-0 skew sensitivity sweep [0.0..1.0] on 8-domain SpMV & HashJoin */
    const double skews[5] = {0.0, 0.40, 0.60, 0.80, 1.00};
    for (int ki = 0; ki < 5; ki++) {
        for (int wi = 0; wi < 2; wi++) {
            int w = (wi == 0) ? 0 : 2; /* SpMV, HashJoin */
            double tb = 0.0, tf = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
            for (int tr = 0; tr < sw_trials; tr++) {
                uint64_t sd = 44449ULL + (uint64_t)(w * 1000 + ki * 59 + tr);
                tb += run_event_driven_sim_ex(traces[w], n_tasks[w], &topo8, POL_UNGUIDED_BASE,
                                              wl_misses[w], 1.0, sd, -1.0, -1, -1.0, skews[ki], 0.45).total_time_ms;
                sim_metrics_t mf = run_event_driven_sim_ex(traces[w], n_tasks[w], &topo8, POL_OTS_FULL,
                                                           wl_misses[w], 1.0, sd, 1.5, 3, 0.80, skews[ki], 0.45);
                tf += mf.total_time_ms; migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
            }
            fprintf(sfp, "node0_skew,%.2f,%s,8,%.4f,%.1f,%.1f,%.1f\n",
                    skews[ki], wl_names[w], tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
        }
    }

    /* 6. M2 Gate Ablation (isolating cooldown vs. hysteresis vs. congestion guard on 8-domain SpMV & BFS) */
    const double g_cfg[5][3] = {
        {1.0, 0.0, 0.99}, /* 0: ungated */
        {1.0, 3.0, 0.99}, /* 1: cooldown_only */
        {1.5, 0.0, 0.99}, /* 2: hysteresis_only */
        {1.0, 0.0, 0.80}, /* 3: guard_only */
        {1.5, 3.0, 0.80}  /* 4: full_gated */
    };
    for (int ci = 0; ci < 5; ci++) {
        for (int wi = 0; wi < 2; wi++) {
            int w = (wi == 0) ? 0 : 1; /* SpMV, BFS */
            double tb = 0.0, tf = 0.0, migs = 0.0, r6 = 0.0, prec = 0.0;
            for (int tr = 0; tr < sw_trials; tr++) {
                uint64_t sd = 33331ULL + (uint64_t)(w * 1000 + ci * 61 + tr);
                tb += run_event_driven_sim(traces[w], n_tasks[w], &topo8, POL_UNGUIDED_BASE, wl_misses[w], 1.0, sd).total_time_ms;
                sim_metrics_t mf = run_event_driven_sim_ex(traces[w], n_tasks[w], &topo8,
                                                           (ci == 0) ? POL_OTS_UNGATED_M2 : POL_OTS_FULL,
                                                           wl_misses[w], 1.0, sd,
                                                           g_cfg[ci][0], (int)g_cfg[ci][1], g_cfg[ci][2], 0.80, 0.45);
                tf += mf.total_time_ms; migs += mf.total_migrations; r6 += mf.re_migs_6ep; prec += mf.mig_precision_pct;
            }
            fprintf(sfp, "gate_ablation,%d,%s,8,%.4f,%.1f,%.1f,%.1f\n",
                    ci, wl_names[w], tb / tf, migs / sw_trials, r6 / sw_trials, prec / sw_trials);
        }
    }
    fclose(sfp);

    for (int w = 0; w < 4; w++) free(traces[w]);
    printf("[02_numa_ots_des_sim] Completed discrete-event simulation (4 workloads x %d trials x %d policies + sweeps).\n",
           n_trials, POL_COUNT);
    return 0;
}
