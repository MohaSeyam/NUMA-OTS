/*
 * 05_qemu_numa_functional_verify.c — Linux NUMA Topology & move_pages(2) Verifier
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Description:
 *   Executes on a multi-node Linux system (or 4-node QEMU virtual NUMA guest)
 *   to verify:
 *     1. Linux kernel NUMA topology and ACPI SLIT distance discovery via libnuma.
 *     2. Real move_pages(2) query mode (nodes == NULL) populating the 64-stripe
 *        user-space Page-Home Hash Table (PHHT) for 4 KiB-aligned buffers.
 *     3. Write-weighted depend-locator affinity scoring A(T, s) (M1).
 *     4. Batched physical page migration via move_pages(2) (MPOL_MF_MOVE) across
 *        NUMA domains and in-place PHHT updates (M2).
 *     5. OpenMP tasking semantics across threads distributed over NUMA domains.
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
 *       05_qemu_numa_functional_verify.c -o qemu_numa_verify -lnuma -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include "numa_ots_core.h"

#define PAGE_SIZE_4K   NUMA_OTS_PAGE_SIZE
#define PHHT_STRIPES   64
#define PHHT_SLOTS     1024
#define BATCH_PAGES    64

typedef struct {
    uintptr_t page_base;
    int home_node;
    int epoch;
    int valid;
} phht_entry_t;

typedef struct {
    omp_lock_t lock;
    phht_entry_t slots[PHHT_SLOTS / PHHT_STRIPES];
    char pad[64];
} phht_stripe_t;

static phht_stripe_t g_phht[PHHT_STRIPES];

static void phht_init(void) {
    for (int s = 0; s < PHHT_STRIPES; s++) {
        omp_init_lock(&g_phht[s].lock);
        memset(g_phht[s].slots, 0, sizeof(g_phht[s].slots));
    }
}

static void phht_destroy(void) {
    for (int s = 0; s < PHHT_STRIPES; s++) {
        omp_destroy_lock(&g_phht[s].lock);
    }
}

static inline uintptr_t align_page_base(const void *addr) {
    return numa_ots_pagebase(addr);
}

static void phht_put(uintptr_t page_base, int node, int epoch) {
    uint64_t h = (page_base >> 12) * 2654435761ULL;
    int stripe = (int)(h & (PHHT_STRIPES - 1));
    int cap = PHHT_SLOTS / PHHT_STRIPES;
    int start = (int)((h >> 6) & (unsigned)(cap - 1));

    omp_set_lock(&g_phht[stripe].lock);
    for (int i = 0; i < cap; i++) {
        int idx = (start + i) % cap;
        if (!g_phht[stripe].slots[idx].valid ||
            g_phht[stripe].slots[idx].page_base == page_base) {
            g_phht[stripe].slots[idx].page_base = page_base;
            g_phht[stripe].slots[idx].home_node = node;
            g_phht[stripe].slots[idx].epoch = epoch;
            g_phht[stripe].slots[idx].valid = 1;
            break;
        }
    }
    omp_unset_lock(&g_phht[stripe].lock);
}

static int phht_get(uintptr_t page_base) {
    uint64_t h = (page_base >> 12) * 2654435761ULL;
    int stripe = (int)(h & (PHHT_STRIPES - 1));
    int cap = PHHT_SLOTS / PHHT_STRIPES;
    int start = (int)((h >> 6) & (unsigned)(cap - 1));
    int res = -1;

    omp_set_lock(&g_phht[stripe].lock);
    for (int i = 0; i < cap; i++) {
        int idx = (start + i) % cap;
        if (!g_phht[stripe].slots[idx].valid) break;
        if (g_phht[stripe].slots[idx].page_base == page_base) {
            res = g_phht[stripe].slots[idx].home_node;
            break;
        }
    }
    omp_unset_lock(&g_phht[stripe].lock);
    return res;
}

/* Query Linux kernel page residency via move_pages(0, cnt, pages, NULL, status, 0) */
static int query_and_cache_pages(void **pages, int *status, int count, int epoch) {
    long rc = move_pages(0, (unsigned long)count, pages, NULL, status, 0);
    if (rc != 0) return -1;
    int valid_cnt = 0;
    for (int i = 0; i < count; i++) {
        if (status[i] >= 0) {
            phht_put(align_page_base(pages[i]), status[i], epoch);
            valid_cnt++;
        }
    }
    return valid_cnt;
}

/* Compute M1 write-weighted affinity score A(T, s) across depend locators */
static int compute_m1_target_node(const void **dep_addrs, const int *is_write,
                                  int n_deps, int creator_node, int num_nodes,
                                  int *out_best_score) {
    int scores[64] = {0};
    long q_zero[64] = {0};
    for (int d = 0; d < n_deps; d++) {
        int home = phht_get(align_page_base(dep_addrs[d]));
        if (home >= 0 && home < num_nodes) {
            scores[home] += is_write[d] ? 2 : 1;
        }
    }
    int best_score = -1;
    for (int s = 0; s < num_nodes; s++) {
        if (scores[s] > best_score) best_score = scores[s];
    }
    *out_best_score = best_score;
    return numa_ots_select_m1_target_eq5(scores, q_zero, creator_node, num_nodes);
}

int main(int argc, char **argv) {
    const char *csv_path = (argc > 1) ? argv[1] : "qemu_functional_verify.csv";
    const char *txt_path = (argc > 2) ? argv[2] : "qemu_functional_report.txt";

    if (numa_available() < 0) {
        fprintf(stderr, "ERROR: libnuma reports NUMA is not available on this kernel.\n");
        return 1;
    }

    int num_nodes = numa_num_configured_nodes();
    int num_cpus  = numa_num_configured_cpus();
    phht_init();

    /* 1. Allocate 64 page-aligned 4 KiB pages on Node 0 and fault them in */
    size_t buf_bytes = BATCH_PAGES * PAGE_SIZE_4K;
    uint8_t *buf = (uint8_t *)numa_alloc_onnode(buf_bytes, 0);
    if (!buf) {
        perror("numa_alloc_onnode");
        return 1;
    }
    for (size_t i = 0; i < buf_bytes; i += PAGE_SIZE_4K) {
        buf[i] = (uint8_t)(i / PAGE_SIZE_4K + 1);
    }

    void *page_ptrs[BATCH_PAGES];
    int status_before[BATCH_PAGES];
    int status_after[BATCH_PAGES];
    int target_nodes[BATCH_PAGES];
    for (int i = 0; i < BATCH_PAGES; i++) {
        page_ptrs[i] = (void *)(buf + (size_t)i * PAGE_SIZE_4K);
        status_before[i] = -1;
        status_after[i]  = -1;
    }

    /* 2. Query initial page residency via move_pages(..., NULL, ...) and populate PHHT */
    int cached_init = query_and_cache_pages(page_ptrs, status_before, BATCH_PAGES, 1);
    int on_node0_before = 0;
    for (int i = 0; i < BATCH_PAGES; i++) {
        if (status_before[i] == 0 && phht_get(align_page_base(page_ptrs[i])) == 0) {
            on_node0_before++;
        }
    }

    /* 3. Verify M1 depend-locator scoring before migration (2 read on Node 0, 1 write on Node 0) */
    const void *deps[2] = { page_ptrs[0], page_ptrs[1] };
    int dep_w[2] = { 1, 0 }; /* out (weight 2) + in (weight 1) = score 3 on Node 0 */
    int score_before = 0;
    int m1_target_before = compute_m1_target_node(deps, dep_w, 2, 0, num_nodes, &score_before);

    /* 4. Migrate batch of 64 pages to destination NUMA node via move_pages(MPOL_MF_MOVE) */
    int dest_node = (num_nodes > 1) ? (num_nodes - 1) : 0;
    for (int i = 0; i < BATCH_PAGES; i++) target_nodes[i] = dest_node;

    long mig_rc = move_pages(0, BATCH_PAGES, page_ptrs, target_nodes, status_after, MPOL_MF_MOVE);
    int verify_cnt = query_and_cache_pages(page_ptrs, status_after, BATCH_PAGES, 2);
    int on_dest_after = 0;
    for (int i = 0; i < BATCH_PAGES; i++) {
        if (status_after[i] == dest_node && phht_get(align_page_base(page_ptrs[i])) == dest_node) {
            on_dest_after++;
        }
    }

    /* 5. Re-evaluate M1 depend-locator scoring after M2 migration + PHHT update */
    int score_after = 0;
    int m1_target_after = compute_m1_target_node(deps, dep_w, 2, 0, num_nodes, &score_after);

    /* 6. Execute OpenMP 3.1-5.0 tasking smoke check across threads */
    int a = 0, b = 0, c = 0, final_ok = 0, if0_ok = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task depend(out: a)
            { a = 19; }
            #pragma omp task depend(out: b)
            { b = 23; }
            #pragma omp task depend(in: a, b) depend(out: c)
            { c = a + b; }
            int enc = omp_get_thread_num();
            #pragma omp task if(0) firstprivate(enc) shared(if0_ok)
            {
                if (omp_get_thread_num() == enc) {
                    #pragma omp atomic
                    if0_ok++;
                }
            }
            #pragma omp task final(1) shared(final_ok)
            {
                if (omp_in_final()) {
                    #pragma omp atomic
                    final_ok++;
                }
            }
            #pragma omp taskwait
        }
    }
    int omp_pass = (c == 42 && if0_ok == 1 && final_ok == 1);

    /* Write CSV summary */
    FILE *fc = fopen(csv_path, "w");
    if (!fc) { perror(csv_path); return 1; }
    fprintf(fc, "configured_nodes,configured_cpus,omp_max_threads,batch_pages,"
                "cached_init,on_node0_before,m1_target_before,m1_score_before,"
                "dest_node,move_pages_rc,on_dest_after,m1_target_after,m1_score_after,omp_smoke_pass\n");
    fprintf(fc, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%d,%d,%d,%d\n",
            num_nodes, num_cpus, omp_get_max_threads(), BATCH_PAGES,
            cached_init, on_node0_before, m1_target_before, score_before,
            dest_node, mig_rc, on_dest_after, m1_target_after, score_after, omp_pass);
    fclose(fc);

    /* Write human-readable report including SLIT distance matrix */
    FILE *ft = fopen(txt_path, "w");
    if (!ft) { perror(txt_path); return 1; }
    fprintf(ft, "=== NUMA-OTS Stage 2: QEMU Virtual NUMA Functional Verification Report ===\n");
    fprintf(ft, "Configured NUMA Nodes : %d\n", num_nodes);
    fprintf(ft, "Configured VCPUs      : %d (OpenMP max_threads = %d)\n", num_cpus, omp_get_max_threads());
    fprintf(ft, "\nACPI SLIT Distance Matrix via numa_distance(u, v):\n    ");
    for (int j = 0; j < num_nodes; j++) fprintf(ft, "  N%-2d", j);
    fprintf(ft, "\n");
    for (int i = 0; i < num_nodes; i++) {
        fprintf(ft, "N%-2d:", i);
        for (int j = 0; j < num_nodes; j++) {
            fprintf(ft, " %4d", numa_distance(i, j));
        }
        fprintf(ft, "\n");
    }
    fprintf(ft, "\n[M1 PHHT Residency Query via move_pages(nodes=NULL)]:\n");
    fprintf(ft, "  Pages allocated on Node 0 : %d / %d verified on Node 0 (cached=%d)\n",
            on_node0_before, BATCH_PAGES, cached_init);
    fprintf(ft, "  Pre-migration M1 target   : Node %d (affinity score A = %d)\n",
            m1_target_before, score_before);
    fprintf(ft, "\n[M2 Physical Page Migration via move_pages(MPOL_MF_MOVE)]:\n");
    fprintf(ft, "  Migrated %d pages Node 0 -> Node %d (syscall rc = %ld, verified=%d)\n",
            BATCH_PAGES, dest_node, mig_rc, verify_cnt);
    fprintf(ft, "  Pages verified on Node %d : %d / %d\n",
            dest_node, on_dest_after, BATCH_PAGES);
    fprintf(ft, "  Post-migration M1 target  : Node %d (affinity score A = %d)\n",
            m1_target_after, score_after);
    fprintf(ft, "\n[OpenMP 3.1-5.0 Tasking Smoke Tests]: %s (c=%d, if0=%d, final=%d)\n",
            omp_pass ? "PASS" : "FAIL", c, if0_ok, final_ok);
    fclose(ft);

    printf("[05_qemu_numa_functional_verify] Nodes=%d | Before: %d/%d on N0 (M1->N%d) | "
           "After move_pages: %d/%d on N%d (M1->N%d) | OpenMP=%s\n",
           num_nodes, on_node0_before, BATCH_PAGES, m1_target_before,
           on_dest_after, BATCH_PAGES, dest_node, m1_target_after,
           omp_pass ? "PASS" : "FAIL");

    numa_free(buf, buf_bytes);
    phht_destroy();
    return (on_node0_before == BATCH_PAGES && on_dest_after == BATCH_PAGES && omp_pass) ? 0 : 1;
}
