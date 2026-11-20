/*
 * 06_sniper_microbench_openmp.c — OpenMP Micro-Kernels for Cache Hierarchy & Energy Analysis
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Experimental Control Guarantee (No Confounding Variables):
 *   For every workload (SpMV, BFS, HashJoin, PageRank), the input data arrays
 *   are initialized identically for both "base" and "ots" policies (producing the
 *   exact same numerical checksum and executing the exact same set of (partition, phase)
 *   computations). The ONLY difference between "base" and "ots" is the task-to-worker
 *   scheduling assignment across consecutive phases:
 *     - "ots"  routes each worker consistently to its home partition across phases
 *              (preserving L1-D and local LLC slice cache reuse across phases).
 *     - "base" dispatches tasks through a cyclic unguided schedule that rotates
 *              partitions across phases ((w + ph * 3) % nparts), evicting the
 *              worker's local cache slice each phase.
 *   Each partition's working set (~28-36 KiB) exceeds the 16 KiB L1-D cache while
 *   fitting inside the 64 KiB per-domain LLC slice, whereas the full 8-partition
 *   dataset (~224-288 KiB) exceeds the LLC slice by ~4x.
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp \
 *       06_sniper_microbench_openmp.c -o sniper_microbench -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

static void *xmalloc(size_t sz) {
    void *p = malloc(sz);
    if (!p) { perror("malloc"); exit(1); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { perror("calloc"); exit(1); }
    return p;
}

/* 1. Scaled-down CSR SpMV (~32 KiB per partition across 8 partitions = 256 KiB total) */
static double run_spmv_micro(int is_ots) {
    const int nrows = 2048, nnz_per_row = 8, nparts = 8, phases = 12;
    const int part_rows = nrows / nparts;

    int *row_ptr = (int *)xmalloc((size_t)(nrows + 1) * sizeof(int));
    int *col_idx = (int *)xmalloc((size_t)nrows * (size_t)nnz_per_row * sizeof(int));
    double *val  = (double *)xmalloc((size_t)nrows * (size_t)nnz_per_row * sizeof(double));
    double *x    = (double *)xmalloc((size_t)nrows * sizeof(double));
    double *y    = (double *)xcalloc((size_t)nrows, sizeof(double));

    for (int r = 0; r < nrows; r++) {
        row_ptr[r] = r * nnz_per_row;
        x[r] = 1.0 / (double)(r + 1);
        int dom_base = (r / part_rows) * part_rows;
        for (int k = 0; k < nnz_per_row; k++) {
            uint64_t h = (uint64_t)r * 1315423911ULL + (uint64_t)k * 2654435761ULL;
            col_idx[r * nnz_per_row + k] = dom_base + (int)(h % (uint64_t)part_rows);
            val[r * nnz_per_row + k] = 0.125;
        }
    }
    row_ptr[nrows] = nrows * nnz_per_row;

    #pragma omp parallel num_threads(nparts)
    {
        #pragma omp for schedule(static, 1)
        for (int w = 0; w < nparts; w++) {
            for (int ph = 0; ph < phases; ph++) {
                int part = is_ots ? w : ((w + ph * 3) % nparts);
                int r0 = part * part_rows;
                int r1 = r0 + part_rows;
                for (int r = r0; r < r1; r++) {
                    double s = 0.0;
                    for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
                        s += val[j] * x[col_idx[j]];
                    }
                    #pragma omp atomic
                    y[r] += s;
                }
            }
        }
    }

    double chk = y[0] + y[nrows / 2] + y[nrows - 1];
    free(row_ptr); free(col_idx); free(val); free(x); free(y);
    return chk;
}

/* 2. Scaled-down Level-Synchronous Frontier BFS (~32 KiB per partition across 8 partitions) */
static double run_bfs_micro(int is_ots) {
    const int nverts = 4096, deg = 8, nparts = 8, phases = 12;
    const int part_verts = nverts / nparts;

    int *edges = (int *)xmalloc((size_t)nverts * (size_t)deg * sizeof(int));
    int *visit = (int *)xcalloc((size_t)nverts, sizeof(int));

    for (int u = 0; u < nverts; u++) {
        int dom_base = (u / part_verts) * part_verts;
        for (int k = 0; k < deg; k++) {
            uint64_t h = (uint64_t)u * 11400714819323198485ULL + (uint64_t)k * 2654435761ULL;
            edges[u * deg + k] = dom_base + (int)((h >> 17) % (uint64_t)part_verts);
        }
    }

    long long total_relax = 0;
    #pragma omp parallel num_threads(nparts) reduction(+:total_relax)
    {
        #pragma omp for schedule(static, 1)
        for (int w = 0; w < nparts; w++) {
            for (int ph = 0; ph < phases; ph++) {
                int part = is_ots ? w : ((w + ph * 3) % nparts);
                int u0 = part * part_verts;
                int u1 = u0 + part_verts;
                for (int u = u0; u < u1; u++) {
                    int local_deg = 0;
                    for (int k = 0; k < deg; k++) {
                        int v = edges[u * deg + k];
                        local_deg += (v & 1);
                    }
                    #pragma omp atomic
                    visit[u] += local_deg;
                    total_relax += local_deg;
                }
            }
        }
    }

    double chk = (double)total_relax + (double)visit[0] + (double)visit[nverts - 1];
    free(edges); free(visit);
    return chk;
}

/* 3. Scaled-down Morsel-Driven Hash Join (~32 KiB per partition across 8 partitions) */
static double run_join_micro(int is_ots) {
    const int build_n = 8192, probe_n = 32768, nparts = 8, phases = 12;
    const int part_build = build_n / nparts;
    const int part_probe = probe_n / nparts;
    const uint32_t sub_mask = (uint32_t)(part_build * 2 - 1);

    uint32_t *ht    = (uint32_t *)xcalloc((size_t)build_n * 2, sizeof(uint32_t));
    uint32_t *probe = (uint32_t *)xmalloc((size_t)probe_n * sizeof(uint32_t));

    for (int p = 0; p < nparts; p++) {
        uint32_t *sub_ht = ht + (size_t)p * (size_t)(part_build * 2);
        for (int i = 0; i < part_build; i++) {
            uint32_t k = (uint32_t)(p * part_build + i) * 2654435761U + 1U;
            sub_ht[k & sub_mask] = k;
        }
        for (int i = 0; i < part_probe; i++) {
            int target_idx = p * part_build + ((i * 17) % part_build);
            probe[p * part_probe + i] = (uint32_t)target_idx * 2654435761U + 1U;
        }
    }

    long long total_hits = 0;
    #pragma omp parallel num_threads(nparts) reduction(+:total_hits)
    {
        #pragma omp for schedule(static, 1)
        for (int w = 0; w < nparts; w++) {
            for (int ph = 0; ph < phases; ph++) {
                int part = is_ots ? w : ((w + ph * 3) % nparts);
                uint32_t *sub_ht = ht + (size_t)part * (size_t)(part_build * 2);
                int i0 = part * part_probe;
                int i1 = i0 + part_probe;
                for (int i = i0; i < i1; i++) {
                    uint32_t k = probe[i];
                    if (sub_ht[k & sub_mask] == k) total_hits++;
                }
            }
        }
    }

    free(ht); free(probe);
    return (double)total_hits;
}

/* 4. Scaled-down Iterative PageRank (~32 KiB per partition across 8 partitions) */
static double run_pagerank_micro(int is_ots) {
    const int nverts = 4096, in_deg = 8, nparts = 8, phases = 12;
    const int part_verts = nverts / nparts;
    const double damp = 0.85;

    int *in_nbrs   = (int *)xmalloc((size_t)nverts * (size_t)in_deg * sizeof(int));
    double *pr_cur = (double *)xmalloc((size_t)nverts * sizeof(double));
    double *pr_nxt = (double *)xcalloc((size_t)nverts, sizeof(double));

    for (int v = 0; v < nverts; v++) {
        pr_cur[v] = 1.0 / (double)nverts;
        int dom_base = (v / part_verts) * part_verts;
        for (int k = 0; k < in_deg; k++) {
            uint64_t h = (uint64_t)v * 7046029254386353131ULL + (uint64_t)k * 2654435761ULL;
            in_nbrs[v * in_deg + k] = dom_base + (int)((h >> 15) % (uint64_t)part_verts);
        }
    }

    #pragma omp parallel num_threads(nparts)
    {
        #pragma omp for schedule(static, 1)
        for (int w = 0; w < nparts; w++) {
            for (int ph = 0; ph < phases; ph++) {
                int part = is_ots ? w : ((w + ph * 3) % nparts);
                int v0 = part * part_verts;
                int v1 = v0 + part_verts;
                for (int v = v0; v < v1; v++) {
                    double sum = 0.0;
                    for (int k = 0; k < in_deg; k++) {
                        int u = in_nbrs[v * in_deg + k];
                        sum += pr_cur[u] / (double)in_deg;
                    }
                    double contrib = (1.0 - damp) / (double)nverts + damp * sum;
                    #pragma omp atomic
                    pr_nxt[v] += contrib;
                }
            }
        }
    }

    double chk = pr_nxt[0] + pr_nxt[nverts / 2] + pr_nxt[nverts - 1];
    free(in_nbrs); free(pr_cur); free(pr_nxt);
    return chk;
}

int main(int argc, char **argv) {
    const char *wl  = (argc > 1) ? argv[1] : "spmv";
    const char *pol = (argc > 2) ? argv[2] : "ots";
    int is_ots = (strcmp(pol, "ots") == 0 || strcmp(pol, "numa_ots_full") == 0) ? 1 : 0;

    double chk = 0.0;
    if (strcmp(wl, "spmv") == 0) {
        chk = run_spmv_micro(is_ots);
    } else if (strcmp(wl, "bfs") == 0) {
        chk = run_bfs_micro(is_ots);
    } else if (strcmp(wl, "join") == 0 || strcmp(wl, "hashjoin") == 0) {
        chk = run_join_micro(is_ots);
    } else if (strcmp(wl, "pagerank") == 0) {
        chk = run_pagerank_micro(is_ots);
    } else {
        fprintf(stderr, "Usage: %s <spmv|bfs|join|pagerank> <base|ots>\n", argv[0]);
        return 1;
    }

    printf("[06_sniper_microbench] workload=%s policy=%s threads=%d checksum=%.6f\n",
           wl, is_ots ? "ots" : "base", omp_get_max_threads(), chk);
    return 0;
}
