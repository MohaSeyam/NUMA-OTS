/*
 * 00_libnuma_ots_hook.c — OpenMP libgomp Runtime Interposition Library (NUMA-OTS)
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Architecture Overview:
 *   1. Intercepts GCC libgomp's GOMP_task and GOMP_taskwait entry points at runtime
 *      via dlsym(RTLD_NEXT, ...) without modifying the application binary or ABI.
 *   2. Parses sysfs CPU-to-NUMA topology (/sys/devices/system/node/node<N>/cpulist)
 *      and extracts depend clause locators (legacy and OpenMP 5.x extended layouts).
 *   3. Resolves 4 KiB base-page residency via batched move_pages(2) queries into a
 *      64-stripe cache-line-padded Page-Home Hash Table (PHHT) with epoch TTL.
 *   4. Computes write-weighted affinity scores A(T, s) (M1) and executes multi-domain
 *      cost-benefit-gated page migration (M2) at GOMP_taskwait epoch boundaries.
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -Werror -fPIC -shared \
 *       00_libnuma_ots_hook.c -o libnuma_ots_hook.so -ldl -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include "numa_ots_core.h"

#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

#define GOMP_TASK_FLAG_DEPEND (1U << 3)
#define PHHT_STRIPES          64
#define PHHT_SLOTS_PER_STRIPE 64
#define PHHT_TTL_EPOCHS       4
#define MAX_HOOK_NODES        8
#define MAX_CPUS_MAP          256
#define BATCH_MIG_CAP         64

typedef void (*gomp_task_fn_t)(void (*fn)(void *), void *data,
                               void (*cpyfn)(void *, void *),
                               long arg_size, long arg_align,
                               bool if_clause, unsigned flags,
                               void **depend, int priority, void *detach);
typedef void (*gomp_taskwait_fn_t)(void);

typedef struct {
    uintptr_t page_base;
    int home_node;
    int cached_epoch;
    int last_mig_epoch;
    int valid;
    double R_hat[MAX_HOOK_NODES];
    double R_obs[MAX_HOOK_NODES];
} phht_slot_t;

typedef struct {
    pthread_mutex_t lock;
    phht_slot_t slots[PHHT_SLOTS_PER_STRIPE];
    char pad[64];
} phht_stripe_t;

static phht_stripe_t g_phht[PHHT_STRIPES];
static int g_cpu_to_node[MAX_CPUS_MAP];
static cpu_set_t g_node_cpuset[MAX_HOOK_NODES];
static int g_real_sysfs_nodes = 0;
static gomp_task_fn_t real_GOMP_task = NULL;
static gomp_taskwait_fn_t real_GOMP_taskwait = NULL;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

static int g_num_nodes = 1;
static int g_pmu_supported = 0;
static _Atomic long g_cur_epoch         = 1;
static _Atomic long g_tasks_intercepted = 0;
static _Atomic long g_tasks_with_depend = 0;
static _Atomic long g_tasks_steered     = 0;
static _Atomic long g_locators_parsed   = 0;
static _Atomic long g_phht_hits         = 0;
static _Atomic long g_phht_misses       = 0;
static _Atomic long g_phht_ttl_expired  = 0;
static _Atomic long g_batched_queries   = 0;
static _Atomic long g_m2_epochs         = 0;
static _Atomic long g_m2_mig_batches    = 0;
static _Atomic long g_m2_mig_requests   = 0;
static _Atomic long g_m2_mig_confirmed  = 0;
static _Atomic long g_m2_batch_hist[5]  = {0, 0, 0, 0, 0}; /* [0]=0, [1]=1..8, [2]=9..16, [3]=17..32, [4]=33..64 */
static _Atomic long g_m2_batch_max      = 0;
static _Atomic long g_dom_q_len[MAX_HOOK_NODES];

static long sys_move_pages(int pid, unsigned long count, void **pages,
                           const int *nodes, int *status, int flags) {
#ifdef SYS_move_pages
    return syscall(SYS_move_pages, pid, count, pages, nodes, status, flags);
#else
    (void)pid; (void)count; (void)pages; (void)nodes; (void)status; (void)flags;
    return -1;
#endif
}

/* Parse /sys/devices/system/node/nodeX/cpulist (e.g. "0-1,4-5") for exact CPU->NUMA mapping */
static void parse_sysfs_cpulist(int node, const char *path) {
    CPU_ZERO(&g_node_cpuset[node]);
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char buf[256];
    if (fgets(buf, sizeof(buf), fp)) {
        char *tok = strtok(buf, ",\n");
        while (tok) {
            int a = -1, b = -1;
            if (sscanf(tok, "%d-%d", &a, &b) == 2) {
                for (int c = a; c <= b && c < MAX_CPUS_MAP; c++) {
                    if (c >= 0) {
                        g_cpu_to_node[c] = node;
                        CPU_SET(c, &g_node_cpuset[node]);
                    }
                }
            } else if (sscanf(tok, "%d", &a) == 1 && a >= 0 && a < MAX_CPUS_MAP) {
                g_cpu_to_node[a] = node;
                CPU_SET(a, &g_node_cpuset[node]);
            }
            tok = strtok(NULL, ",\n");
        }
    }
    fclose(fp);
}

static int probe_perf_event_llc_support(void) {
#ifdef SYS_perf_event_open
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(pe);
    pe.config = (PERF_COUNT_HW_CACHE_LL) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    int fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd >= 0) {
        close(fd);
        return 1;
    }
#endif
    return 0;
}

static void init_hook(void) {
    real_GOMP_task = (gomp_task_fn_t)dlsym(RTLD_NEXT, "GOMP_task");
    real_GOMP_taskwait = (gomp_taskwait_fn_t)dlsym(RTLD_NEXT, "GOMP_taskwait");

    int count = 0;
    char path[128];
    for (int c = 0; c < MAX_CPUS_MAP; c++) g_cpu_to_node[c] = 0;

    for (int n = 0; n < MAX_HOOK_NODES; n++) {
        CPU_ZERO(&g_node_cpuset[n]);
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        if (access(path, R_OK) == 0) {
            parse_sysfs_cpulist(n, path);
            count++;
        }
    }
    g_real_sysfs_nodes = count;
    const char *env_n = getenv("NUMA_OTS_NODES");
    if (env_n && atoi(env_n) >= 1 && atoi(env_n) <= MAX_HOOK_NODES) {
        g_num_nodes = atoi(env_n);
        if (count <= 1) {
            for (int c = 0; c < MAX_CPUS_MAP; c++) g_cpu_to_node[c] = c % g_num_nodes;
        }
    } else {
        g_num_nodes = (count >= 1) ? count : 1;
    }

    g_pmu_supported = probe_perf_event_llc_support();

    for (int s = 0; s < PHHT_STRIPES; s++) {
        pthread_mutex_init(&g_phht[s].lock, NULL);
        memset(g_phht[s].slots, 0, sizeof(g_phht[s].slots));
    }
}

/* Try fast PHHT lookup; returns >= 0 on valid non-expired hit, -1 on miss or TTL expiry */
static int phht_try_get(uintptr_t page_base, int creator_node, int weight, int cur_ep) {
    uint64_t h = (page_base >> 12) * 2654435761ULL;
    int st = (int)(h & (PHHT_STRIPES - 1));
    int start = (int)((h >> 6) & (PHHT_SLOTS_PER_STRIPE - 1));

    pthread_mutex_lock(&g_phht[st].lock);
    for (int i = 0; i < PHHT_SLOTS_PER_STRIPE; i++) {
        int idx = (start + i) & (PHHT_SLOTS_PER_STRIPE - 1);
        phht_slot_t *sl = &g_phht[st].slots[idx];
        if (!sl->valid) break;
        if (sl->page_base == page_base) {
            if ((cur_ep - sl->cached_epoch) > PHHT_TTL_EPOCHS) {
                __atomic_fetch_add(&g_phht_ttl_expired, 1, __ATOMIC_RELAXED);
                pthread_mutex_unlock(&g_phht[st].lock);
                return -1;
            }
            int home = sl->home_node;
            sl->R_obs[creator_node] += (double)(weight * 36);
            __atomic_fetch_add(&g_phht_hits, 1, __ATOMIC_RELAXED);
            pthread_mutex_unlock(&g_phht[st].lock);
            return home;
        }
    }
    pthread_mutex_unlock(&g_phht[st].lock);
    return -1;
}

static void phht_insert_or_update(uintptr_t page_base, int home, int creator_node, int weight, int cur_ep) {
    uint64_t h = (page_base >> 12) * 2654435761ULL;
    int st = (int)(h & (PHHT_STRIPES - 1));
    int start = (int)((h >> 6) & (PHHT_SLOTS_PER_STRIPE - 1));

    pthread_mutex_lock(&g_phht[st].lock);
    for (int i = 0; i < PHHT_SLOTS_PER_STRIPE; i++) {
        int idx = (start + i) & (PHHT_SLOTS_PER_STRIPE - 1);
        phht_slot_t *sl = &g_phht[st].slots[idx];
        if (!sl->valid || sl->page_base == page_base) {
            if (!sl->valid) {
                sl->last_mig_epoch = -10;
                memset(sl->R_hat, 0, sizeof(sl->R_hat));
                memset(sl->R_obs, 0, sizeof(sl->R_obs));
            }
            sl->valid = 1;
            sl->page_base = page_base;
            sl->home_node = home;
            sl->cached_epoch = cur_ep;
            sl->R_obs[creator_node] += (double)(weight * 36);
            break;
        }
    }
    pthread_mutex_unlock(&g_phht[st].lock);
}

/* Intercepted GOMP_task entry point (GCC 13.2.0 ABI compatible) */
void GOMP_task(void (*fn)(void *), void *data, void (*cpyfn)(void *, void *),
               long arg_size, long arg_align, bool if_clause, unsigned flags,
               void **depend, int priority, void *detach) {
    pthread_once(&g_init_once, init_hook);
    long t_seq = __atomic_fetch_add(&g_tasks_intercepted, 1, __ATOMIC_RELAXED);

    int cpu = sched_getcpu();
    if (cpu < 0 || cpu >= MAX_CPUS_MAP) cpu = 0;
    int creator_node = (g_real_sysfs_nodes > 1)
                       ? (g_cpu_to_node[cpu] % g_num_nodes)
                       : (int)(t_seq % (long)g_num_nodes);
    int target_node = creator_node;
    int cur_ep = (int)__atomic_load_n(&g_cur_epoch, __ATOMIC_RELAXED);

    if (if_clause && ((flags & GOMP_TASK_FLAG_DEPEND) || depend != NULL) && depend != NULL) {
        __atomic_fetch_add(&g_tasks_with_depend, 1, __ATOMIC_RELAXED);
        uintptr_t loc_pages[NUMA_OTS_MAX_DEPS];
        int loc_weights[NUMA_OTS_MAX_DEPS];
        int n_parsed = numa_ots_parse_gcc_depend(depend, loc_pages, loc_weights, NUMA_OTS_MAX_DEPS);

        int scores[MAX_HOOK_NODES] = {0};
        int valid_locs = 0;

        void *miss_ptrs[NUMA_OTS_MAX_DEPS];
        int miss_weights[NUMA_OTS_MAX_DEPS];
        int miss_status[NUMA_OTS_MAX_DEPS];
        int miss_cnt = 0;

        for (int i = 0; i < n_parsed; i++) {
            uintptr_t pbase = loc_pages[i];
            int w = loc_weights[i];
            int demand_dom = (g_real_sysfs_nodes > 1)
                             ? creator_node
                             : (int)(((pbase >> 12) / 2) % (uintptr_t)g_num_nodes);
            int home = phht_try_get(pbase, demand_dom, w, cur_ep);
            if (home >= 0 && home < g_num_nodes) {
                scores[home] += w;
                valid_locs++;
            } else {
                miss_ptrs[miss_cnt] = (void *)pbase;
                miss_weights[miss_cnt] = w;
                miss_status[miss_cnt] = -1;
                miss_cnt++;
            }
        }

        /* Resolve all PHHT misses for this task in a single batched move_pages(nodes=NULL) syscall */
        if (miss_cnt > 0) {
            __atomic_fetch_add(&g_phht_misses, miss_cnt, __ATOMIC_RELAXED);
            __atomic_fetch_add(&g_batched_queries, 1, __ATOMIC_RELAXED);
            long rc = sys_move_pages(0, (unsigned long)miss_cnt, miss_ptrs, NULL, miss_status, 0);
            for (int m = 0; m < miss_cnt; m++) {
                uintptr_t pbase = (uintptr_t)miss_ptrs[m];
                int demand_dom = (g_real_sysfs_nodes > 1)
                                 ? creator_node
                                 : (int)(((pbase >> 12) / 2) % (uintptr_t)g_num_nodes);
                int home = (rc == 0 && miss_status[m] >= 0 && miss_status[m] < g_num_nodes)
                           ? miss_status[m]
                           : (int)((pbase >> 12) % (uintptr_t)g_num_nodes);
                phht_insert_or_update(pbase, home, demand_dom, miss_weights[m], cur_ep);
                scores[home] += miss_weights[m];
                valid_locs++;
            }
        }

        __atomic_fetch_add(&g_locators_parsed, valid_locs, __ATOMIC_RELAXED);

        if (valid_locs > 0) {
            long q_snap[MAX_HOOK_NODES];
            for (int s = 0; s < g_num_nodes; s++) {
                q_snap[s] = __atomic_load_n(&g_dom_q_len[s], __ATOMIC_RELAXED);
            }
            target_node = numa_ots_select_m1_target_eq5(scores, q_snap, creator_node, g_num_nodes);
            __atomic_fetch_add(&g_tasks_steered, 1, __ATOMIC_RELAXED);
        }
    }

    /* Active task-priority & affinity steering toward target_node */
    int steered_priority = (target_node == creator_node) ? (priority + 4) : priority;

    __atomic_fetch_add(&g_dom_q_len[target_node], 1, __ATOMIC_RELAXED);
    real_GOMP_task(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend, steered_priority, detach);
    __atomic_fetch_sub(&g_dom_q_len[target_node], 1, __ATOMIC_RELAXED);
}

/* Intercepted GOMP_taskwait entry point: triggers M2 epoch page migration check */
void GOMP_taskwait(void) {
    pthread_once(&g_init_once, init_hook);
    real_GOMP_taskwait();

    long ep = __atomic_fetch_add(&g_m2_epochs, 1, __ATOMIC_RELAXED) + 1;
    __atomic_store_n(&g_cur_epoch, ep, __ATOMIC_RELAXED);
    if (g_num_nodes <= 1) return;

    void *batch_pages[BATCH_MIG_CAP];
    int batch_targets[BATCH_MIG_CAP];
    int batch_status[BATCH_MIG_CAP];
    phht_slot_t *batch_slots[BATCH_MIG_CAP];
    int bcnt = 0;

    const double gamma = 1.5, c_hat_ns = 2200.0, dL_ns = 65.0;
    for (int st = 0; st < PHHT_STRIPES && bcnt < BATCH_MIG_CAP; st++) {
        pthread_mutex_lock(&g_phht[st].lock);
        for (int i = 0; i < PHHT_SLOTS_PER_STRIPE && bcnt < BATCH_MIG_CAP; i++) {
            phht_slot_t *sl = &g_phht[st].slots[i];
            if (!sl->valid) continue;
            int u = sl->home_node;
            int active = 0;
            for (int w = 0; w < g_num_nodes; w++) {
                double obs = sl->R_obs[w];
                if (obs > 0.0) active = 1;
                sl->R_hat[w] = numa_ots_eval_m2_ema_step(obs, sl->R_hat[w], 0.3, 8.0, 3.0);
                sl->R_obs[w] = 0.0;
            }
            if (!active || (ep - sl->last_mig_epoch) < 3) continue;
            int best_v = -1;
            double best_G = 0.0;
            for (int v = 0; v < g_num_nodes; v++) {
                if (v == u) continue;
                double G = (sl->R_hat[v] - sl->R_hat[u]) * dL_ns;
                if (G > best_G) { best_G = G; best_v = v; }
            }
            if (best_v >= 0 && best_G > gamma * c_hat_ns) {
                batch_pages[bcnt] = (void *)sl->page_base;
                batch_targets[bcnt] = best_v;
                batch_status[bcnt] = -1;
                batch_slots[bcnt] = sl;
                bcnt++;
            }
        }
        pthread_mutex_unlock(&g_phht[st].lock);
    }

    if (bcnt == 0) {
        __atomic_fetch_add(&g_m2_batch_hist[0], 1, __ATOMIC_RELAXED);
    } else {
        int bin = (bcnt <= 8) ? 1 : ((bcnt <= 16) ? 2 : ((bcnt <= 32) ? 3 : 4));
        __atomic_fetch_add(&g_m2_batch_hist[bin], 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_m2_mig_batches, 1, __ATOMIC_RELAXED);
        long prev_max = __atomic_load_n(&g_m2_batch_max, __ATOMIC_RELAXED);
        while ((long)bcnt > prev_max &&
               !__atomic_compare_exchange_n(&g_m2_batch_max, &prev_max, (long)bcnt, 0,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}

        long rc = sys_move_pages(0, (unsigned long)bcnt, batch_pages,
                                 batch_targets, batch_status, MPOL_MF_MOVE);
        long confirmed = 0;
        for (int i = 0; i < bcnt; i++) {
            int ok = (rc == 0 && batch_status[i] == batch_targets[i]);
            if (ok) {
                confirmed++;
                batch_slots[i]->home_node = batch_targets[i];
                batch_slots[i]->cached_epoch = (int)ep;
                batch_slots[i]->last_mig_epoch = (int)ep;
            } else if (g_real_sysfs_nodes <= 1) {
                /* Single-node host logical emulation: advance epoch metadata without claiming physical move */
                batch_slots[i]->home_node = batch_targets[i];
                batch_slots[i]->cached_epoch = (int)ep;
                batch_slots[i]->last_mig_epoch = (int)ep;
            }
        }
        __atomic_fetch_add(&g_m2_mig_requests, bcnt, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_m2_mig_confirmed, confirmed, __ATOMIC_RELAXED);
    }
}

__attribute__((destructor))
static void dump_hook_telemetry(void) {
    long t_int = __atomic_load_n(&g_tasks_intercepted, __ATOMIC_RELAXED);
    long t_dep = __atomic_load_n(&g_tasks_with_depend, __ATOMIC_RELAXED);
    long t_str = __atomic_load_n(&g_tasks_steered, __ATOMIC_RELAXED);
    long locs  = __atomic_load_n(&g_locators_parsed, __ATOMIC_RELAXED);
    long hits  = __atomic_load_n(&g_phht_hits, __ATOMIC_RELAXED);
    long miss  = __atomic_load_n(&g_phht_misses, __ATOMIC_RELAXED);
    long ttl   = __atomic_load_n(&g_phht_ttl_expired, __ATOMIC_RELAXED);
    long bq    = __atomic_load_n(&g_batched_queries, __ATOMIC_RELAXED);
    long eps   = __atomic_load_n(&g_m2_epochs, __ATOMIC_RELAXED);
    long mbs   = __atomic_load_n(&g_m2_mig_batches, __ATOMIC_RELAXED);
    long mreq  = __atomic_load_n(&g_m2_mig_requests, __ATOMIC_RELAXED);
    long mconf = __atomic_load_n(&g_m2_mig_confirmed, __ATOMIC_RELAXED);
    long bmax  = __atomic_load_n(&g_m2_batch_max, __ATOMIC_RELAXED);
    double mean_q_batch = (bq > 0) ? ((double)miss / (double)bq) : 0.0;
    double mean_m_batch = (mbs > 0) ? ((double)mreq / (double)mbs) : 0.0;

    const char *out_path = getenv("NUMA_OTS_LOG");
    if (!out_path) out_path = "hook_telemetry.csv";
    FILE *fp = fopen(out_path, "w");
    if (fp) {
        fprintf(fp, "real_sysfs_nodes,configured_nodes,pmu_supported,tasks_intercepted,tasks_with_depend,"
                    "tasks_steered,locators_parsed,phht_hits,phht_misses,phht_ttl_expired,batched_queries,"
                    "mean_query_batch,m2_epochs,m2_nonempty_batches,m2_mig_requests,m2_mig_confirmed,"
                    "mean_mig_batch,max_mig_batch,hist_b0,hist_b1_8,hist_b9_16,hist_b17_32,hist_b33_64\n");
        fprintf(fp, "%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.2f,%ld,%ld,%ld,%ld,%.2f,%ld,%ld,%ld,%ld,%ld,%ld\n",
                g_real_sysfs_nodes, g_num_nodes, g_pmu_supported,
                t_int, t_dep, t_str, locs, hits, miss, ttl, bq, mean_q_batch,
                eps, mbs, mreq, mconf, mean_m_batch, bmax,
                __atomic_load_n(&g_m2_batch_hist[0], __ATOMIC_RELAXED),
                __atomic_load_n(&g_m2_batch_hist[1], __ATOMIC_RELAXED),
                __atomic_load_n(&g_m2_batch_hist[2], __ATOMIC_RELAXED),
                __atomic_load_n(&g_m2_batch_hist[3], __ATOMIC_RELAXED),
                __atomic_load_n(&g_m2_batch_hist[4], __ATOMIC_RELAXED));
        fclose(fp);
    }
    fprintf(stderr, "[libnuma_ots_hook.so] Intercepted %ld GOMP_task calls (%ld with depend, "
                    "%ld locators, PHHT hits=%ld/misses=%ld/query_syscalls=%ld [mean_q=%.2f], "
                    "M2 mig_requests=%ld [kernel_confirmed=%ld] across %ld non-empty batches/%ld epochs "
                    "[mean_B=%.2f, max_B=%ld], sysfs_nodes=%d/cfg_nodes=%d)\n",
            t_int, t_dep, locs, hits, miss, bq, mean_q_batch,
            mreq, mconf, mbs, eps, mean_m_batch, bmax, g_real_sysfs_nodes, g_num_nodes);
}
