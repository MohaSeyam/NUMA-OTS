/*
 * numa_ots_core.h — Shared Core Algorithms for NUMA-OTS Runtime Hook, DES Simulator & Unit Tests
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Centralizes the production implementations of:
 *   1. GCC libgomp GOMP_task depend array parser (legacy & OpenMP 5.x extended layouts)
 *   2. M1 Equation (5) affinity scoring and tie-breaking (creator preservation + shortest queue)
 *   3. M2 Equation (7) stabilized EMA demand predictor with step-change reset guard
 *   4. M2 Proposition 1 analytical break-even demand threshold R_min
 */

#ifndef NUMA_OTS_CORE_H
#define NUMA_OTS_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define NUMA_OTS_PAGE_SIZE 4096UL
#define NUMA_OTS_MAX_DEPS  16

static inline uintptr_t numa_ots_pagebase(const void *addr) {
    return ((uintptr_t)addr) & ~(NUMA_OTS_PAGE_SIZE - 1UL);
}

/*
 * Parse GCC 13.2-15.2 libgomp GOMP_task depend array:
 *   - Legacy format (depend[0] != 0):
 *       depend[0] = total count, depend[1] = n_out, depend[2..] = locator pointers
 *   - Extended OpenMP 5.x format (depend[0] == 0):
 *       depend[1] = total count, depend[2] = n_out, depend[3] = n_mutexinoutset,
 *       depend[4] = n_in, depend[5..] = [out/inout, mutexinoutset, in, inoutset]
 *   Skips omp_all_memory tokens (NULL), aligns locators to 4 KiB base pages (Eq. 3),
 *   and deduplicates into P(T) with w(p) = max_{a_i in p} w_dep(a_i) (Eq. 4).
 */
static inline int numa_ots_parse_gcc_depend(void **depend,
                                            uintptr_t *out_page_bases,
                                            int *out_weights,
                                            int max_out) {
    if (!depend || !out_page_bases || !out_weights || max_out <= 0) return 0;
    uintptr_t tag0 = (uintptr_t)depend[0];
    size_t n_total = 0, n_out = 0, n_mt = 0, n_in = 0, first_idx = 2;
    int is_ext = 0;

    if (tag0 != 0) {
        n_total = (size_t)tag0;
        n_out = (size_t)(uintptr_t)depend[1];
        first_idx = 2;
    } else {
        is_ext = 1;
        n_total = (size_t)(uintptr_t)depend[1];
        n_out = (size_t)(uintptr_t)depend[2];
        n_mt  = (size_t)(uintptr_t)depend[3];
        n_in  = (size_t)(uintptr_t)depend[4];
        first_idx = 5;
    }

    if (n_total > 64) n_total = 64;
    int valid_cnt = 0;
    for (size_t i = 0; i < n_total; i++) {
        void *loc = depend[first_idx + i];
        if (loc == NULL) continue; /* omp_all_memory serialization token */
        uintptr_t pbase = numa_ots_pagebase(loc);
        int w = 1;
        if (!is_ext) {
            w = (i < n_out) ? 2 : 1;
        } else {
            /* [0..n_out+n_mt-1] = out/inout/mutexinoutset (w=2),
               [n_out+n_mt..n_out+n_mt+n_in-1] = in (w=1),
               [n_out+n_mt+n_in..n_total-1] = inoutset (w=2) */
            if (i < n_out + n_mt || i >= n_out + n_mt + n_in) w = 2;
            else w = 1;
        }
        int found = 0;
        for (int j = 0; j < valid_cnt; j++) {
            if (out_page_bases[j] == pbase) {
                if (w > out_weights[j]) out_weights[j] = w;
                found = 1;
                break;
            }
        }
        if (!found && valid_cnt < max_out) {
            out_page_bases[valid_cnt] = pbase;
            out_weights[valid_cnt] = w;
            valid_cnt++;
        }
    }
    return valid_cnt;
}

/*
 * M1 Target Domain Selection — Equation (5):
 *   Given domain affinity scores[0..N-1], domain queue lengths q_len[0..N-1],
 *   and creator_node:
 *     - Let S* = argmax_s scores[s].
 *     - If creator_node in S*, return creator_node (preserves L2/LLC locality).
 *     - Otherwise return argmin_{s in S*} q_len[s] (avoids Node-0 hotspot bias).
 */
static inline int numa_ots_select_m1_target_eq5(const int *scores,
                                                const long *q_len,
                                                int creator_node,
                                                int num_nodes) {
    int max_sc = -1;
    for (int s = 0; s < num_nodes; s++) {
        if (scores[s] > max_sc) max_sc = scores[s];
    }
    if (creator_node >= 0 && creator_node < num_nodes && scores[creator_node] == max_sc) {
        return creator_node;
    }
    int best_s = 0;
    long min_q = INT64_MAX;
    for (int s = 0; s < num_nodes; s++) {
        if (scores[s] == max_sc && q_len[s] < min_q) {
            min_q = q_len[s];
            best_s = s;
        }
    }
    return best_s;
}

/*
 * M2 Equation (7) Stabilized EMA Demand Predictor with Step-Change Guard:
 *   Resets R_hat <- R_obs when |R_obs - R_prev| / max(R_prev, eps) > reset_ratio.
 */
static inline double numa_ots_eval_m2_ema_step(double r_obs,
                                               double r_prev,
                                               double beta_ema,
                                               double eps_guard,
                                               double reset_ratio) {
    double denom = (r_prev > eps_guard) ? r_prev : eps_guard;
    if (fabs(r_obs - r_prev) / denom > reset_ratio) {
        return r_obs;
    }
    return beta_ema * r_obs + (1.0 - beta_ema) * r_prev;
}

/*
 * M2 Proposition 1 Analytical Break-Even Demand Threshold R_min — Equation (11):
 *   R_min = floor(gamma * C_hat_ns / delta_L_ns) + 1
 */
static inline int numa_ots_calc_rmin_prop1(double gamma,
                                           double c_hat_ns,
                                           double delta_l_ns) {
    if (delta_l_ns <= 0.0) return 1000000000;
    return (int)floor((gamma * c_hat_ns) / delta_l_ns) + 1;
}

#endif /* NUMA_OTS_CORE_H */
