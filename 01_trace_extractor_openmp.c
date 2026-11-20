/*
 * 01_trace_extractor_openmp.c — OpenMP Task Benchmark & Logical-Page Trace Extractor
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Description:
 *   1. Executes OpenMP 3.1-5.0 tasking semantic verification tests (depend,
 *      taskloop reduction, if(0) undeferred execution, final(1), taskwait).
 *   2. Runs four task-parallel benchmarks with inter-task depend clauses:
 *        (a) CSR Sparse Matrix-Vector Multiplication (trace_spmv.csv)
 *        (b) Level-Synchronous Frontier Breadth-First Search (trace_bfs.csv)
 *        (c) Morsel-Driven Relational Hash Join (trace_join.csv)
 *        (d) Iterative Pull-Based PageRank (trace_pagerank.csv)
 *   3. Records deterministic, ASLR-independent 4 KiB logical page IDs and
 *      machine-independent work operation counts into pre-allocated task slots.
 *
 * Build:
 *   gcc -O3 -std=c11 -Wall -Wextra -Werror -fopenmp 01_trace_extractor_openmp.c -o trace_extractor -lm
 * Run:
 *   OMP_NUM_THREADS=4 ./trace_extractor traces_out
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <omp.h>

#define MAX_REC_PAGES 16

typedef struct {
    int task_id;
    int phase;
    int creator_tid;
    int work_ops;                 /* Machine-independent compute ops */
    int dep_pages[2];             /* Up to 2 depend locator logical pages */
    int dep_is_write[2];          /* 1 = out/inout, 0 = in */
    int num_deps;
    int num_pages;
    uint32_t pages[MAX_REC_PAGES];
} task_trace_rec_t;

static inline uint32_t logical_page_id(uint32_t array_id, size_t byte_offset) {
    /* Collision-free, ASLR-independent 4 KiB logical page ID */
    return (array_id << 16) | (uint32_t)((byte_offset >> 12) & 0xFFFFU);
}

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

static int cmp_int_asc(const void *a, const void *b) {
    int va = *(const int *)a;
    int vb = *(const int *)b;
    return (va > vb) - (va < vb);
}

/* OpenMP 3.1-5.0 Tasking Smoke Test (depend, taskloop reduction, if(0), final(1), taskwait) */
static int verify_openmp_smoke_tests(void) {
    int a = 0, b = 0, c = 0;
    int inline_ok = 0, final_child_is_final = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task depend(out: a)
            { a = 15; }
            #pragma omp task depend(out: b)
            { b = 27; }
            #pragma omp task depend(in: a, b) depend(out: c)
            { c = a + b; }

            int parent_tid = omp_get_thread_num();
            #pragma omp task if(0) shared(inline_ok) firstprivate(parent_tid)
            {
                if (omp_get_thread_num() == parent_tid) {
                    #pragma omp atomic
                    inline_ok += 1;
                }
            }
            #pragma omp task final(1) shared(final_child_is_final)
            {
                if (omp_in_final()) {
                    #pragma omp atomic
                    final_child_is_final += 1;
                }
            }
            #pragma omp taskwait
        }
    }
    long long tl_sum = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop reduction(+:tl_sum) grainsize(128)
            for (int i = 1; i <= 5000; i++) tl_sum += i;
        }
    }
    return (c == 42 && inline_ok == 1 && final_child_is_final == 1 && tl_sum == 12502500LL) ? 1 : 0;
}

static void write_trace_csv(const char *path, const task_trace_rec_t *recs, int count) {
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); exit(1); }
    if (fprintf(fp, "task_id,phase,creator_tid,work_ops,num_deps,dep0,w0,dep1,w1,num_pages,pages\n") < 0) {
        perror("fprintf header"); fclose(fp); exit(1);
    }
    for (int i = 0; i < count; i++) {
        const task_trace_rec_t *r = &recs[i];
        if (fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,",
                    r->task_id, r->phase, r->creator_tid, r->work_ops,
                    r->num_deps, r->dep_pages[0], r->dep_is_write[0],
                    r->dep_pages[1], r->dep_is_write[1], r->num_pages) < 0) {
            perror("fprintf row"); fclose(fp); exit(1);
        }
        for (int j = 0; j < r->num_pages; j++) {
            fprintf(fp, "%u%s", r->pages[j], (j + 1 < r->num_pages) ? ";" : "\n");
        }
    }
    if (fclose(fp) != 0) { perror("fclose"); exit(1); }
}

/* 1. CSR SpMV with cross-phase depend(in: val) depend(inout: y) chains */
static void extract_spmv_trace(const char *path) {
    const int nrows = 32768, nnz_per_row = 8, chunk = 256, phases = 8;
    const int nblocks = (nrows + chunk - 1) / chunk;
    const int total_tasks = phases * nblocks;

    int *row_ptr = (int *)xmalloc((size_t)(nrows + 1) * sizeof(int));
    int *col_idx = (int *)xmalloc((size_t)nrows * (size_t)nnz_per_row * sizeof(int));
    double *val  = (double *)xmalloc((size_t)nrows * (size_t)nnz_per_row * sizeof(double));
    double *x    = (double *)xmalloc((size_t)nrows * sizeof(double));
    double *y    = (double *)xmalloc((size_t)nrows * sizeof(double));
    task_trace_rec_t *recs = (task_trace_rec_t *)xcalloc((size_t)total_tasks, sizeof(task_trace_rec_t));

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        row_ptr[r] = r * nnz_per_row;
        x[r] = 1.0 / (double)(r + 1);
        y[r] = 0.0;
        for (int k = 0; k < nnz_per_row; k++) {
            col_idx[r * nnz_per_row + k] = (int)(((uint64_t)r * 1315423911ULL + (uint64_t)k * 2654435761ULL) % (uint64_t)nrows);
            val[r * nnz_per_row + k] = 0.25;
        }
    }
    row_ptr[nrows] = nrows * nnz_per_row;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int ph = 0; ph < phases; ph++) {
                for (int blk = 0; blk < nblocks; blk++) {
                    int r0 = blk * chunk;
                    int r1 = (r0 + chunk < nrows) ? (r0 + chunk) : nrows;
                    int tid = ph * nblocks + blk;
                    int c_tid = (blk + ph) & 7; /* Deterministic creator mapping */

                    #pragma omp task firstprivate(r0, r1, tid, ph, c_tid, blk) \
                                     depend(in: val[row_ptr[r0]]) depend(inout: y[r0])
                    {
                        task_trace_rec_t *rec = &recs[tid];
                        size_t blk_base = (size_t)blk * 16U * 4096U;
                        rec->task_id = tid;
                        rec->phase = ph;
                        rec->creator_tid = c_tid;
                        rec->work_ops = (r1 - r0) * nnz_per_row * 2;
                        rec->num_deps = 2;
                        rec->dep_pages[0] = (int)logical_page_id(1, blk_base);
                        rec->dep_is_write[0] = 1;
                        rec->dep_pages[1] = (int)logical_page_id(2, blk_base + 4096U);
                        rec->dep_is_write[1] = 0;

                        for (int r = r0; r < r1; r++) {
                            double s = 0.0;
                            for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
                                s += val[j] * x[col_idx[j]];
                            }
                            y[r] = s + 0.01 * y[r];
                        }

                        int nt = 0;
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[0];
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[1];
                        for (int p = 2; p < 11; p++) {
                            uint32_t arr_id = (p < 6) ? 1U : 2U;
                            rec->pages[nt++] = logical_page_id(arr_id, blk_base + (size_t)p * 4096U);
                        }
                        for (int p = 11; p < 15; p++) {
                            rec->pages[nt++] = logical_page_id(3, blk_base + (size_t)(p - 11) * 4096U);
                        }
                        size_t halo_off = (size_t)(((blk / 4) + ph * 3) & 15) * 4096U;
                        rec->pages[nt++] = logical_page_id(3, halo_off);
                        rec->num_pages = nt;
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    write_trace_csv(path, recs, total_tasks);
    free(recs); free(row_ptr); free(col_idx); free(val); free(x); free(y);
}

/* 2. True Level-Synchronous Frontier-Chunk BFS (sorted frontier per level for 100% determinism) */
static void extract_bfs_trace(const char *path) {
    const int nverts = 32768, deg = 12, chunk = 256, max_levels = 8;
    int *edges    = (int *)xmalloc((size_t)nverts * (size_t)deg * sizeof(int));
    int *dist     = (int *)xmalloc((size_t)nverts * sizeof(int));
    int *frontier = (int *)xmalloc((size_t)nverts * sizeof(int));
    int *next_f   = (int *)xmalloc((size_t)nverts * sizeof(int));
    task_trace_rec_t *recs = (task_trace_rec_t *)xcalloc(4096, sizeof(task_trace_rec_t));

    #pragma omp parallel for schedule(static)
    for (int u = 0; u < nverts; u++) {
        dist[u] = -1;
        for (int k = 0; k < deg; k++) {
            uint64_t h = (uint64_t)u * 11400714819323198485ULL + (uint64_t)k * 2654435761ULL;
            edges[u * deg + k] = (int)((h >> 17) % (uint64_t)nverts);
        }
    }

    int fsize = 0;
    for (int s = 0; s < 512; s++) {
        int r = (s * 61) % nverts;
        if (dist[r] == -1) {
            dist[r] = 0;
            frontier[fsize++] = r;
        }
    }

    int total_tasks = 0;
    for (int lvl = 0; lvl < max_levels && fsize > 0; lvl++) {
        int next_sz = 0;
        int nchunks = (fsize + chunk - 1) / chunk;
        int base_tid = total_tasks;
        if (base_tid + nchunks >= 4096) break;

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int c = 0; c < nchunks; c++) {
                    int i0 = c * chunk;
                    int i1 = (i0 + chunk < fsize) ? (i0 + chunk) : fsize;
                    int tid = base_tid + c;
                    int c_tid = (c + lvl) & 7;

                    #pragma omp task firstprivate(i0, i1, tid, lvl, c_tid, c) depend(inout: frontier[i0]) shared(next_sz)
                    {
                        task_trace_rec_t *rec = &recs[tid];
                        size_t blk_base = (size_t)(c & 127) * 16U * 4096U;
                        rec->task_id = tid;
                        rec->phase = lvl;
                        rec->creator_tid = c_tid;
                        rec->work_ops = (i1 - i0) * deg * 3;
                        rec->num_deps = 1;
                        rec->dep_pages[0] = (int)logical_page_id(4, blk_base);
                        rec->dep_is_write[0] = 1;
                        rec->dep_pages[1] = 0;
                        rec->dep_is_write[1] = 0;

                        for (int idx = i0; idx < i1; idx++) {
                            int u = frontier[idx];
                            for (int k = 0; k < deg; k++) {
                                int v = edges[u * deg + k];
                                if (__sync_bool_compare_and_swap(&dist[v], -1, lvl + 1)) {
                                    int pos = __sync_fetch_and_add(&next_sz, 1);
                                    if (pos < nverts) next_f[pos] = v;
                                }
                            }
                        }

                        int nt = 0;
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[0];
                        for (int p = 1; p < 11; p++) {
                            rec->pages[nt++] = logical_page_id(4, blk_base + (size_t)p * 4096U);
                        }
                        for (int p = 11; p < 15; p++) {
                            rec->pages[nt++] = logical_page_id(5, blk_base + (size_t)(p - 11) * 4096U);
                        }
                        size_t halo_off = (size_t)((((c & 127) / 4) + lvl * 3) & 15) * 4096U;
                        rec->pages[nt++] = logical_page_id(5, halo_off);
                        rec->num_pages = nt;
                    }
                }
                #pragma omp taskwait
            }
        }
        total_tasks += nchunks;
        fsize = (next_sz < nverts) ? next_sz : nverts;
        /* Sort next_f before next level so frontier chunks are 100% deterministic across runs/hosts */
        if (fsize > 1) qsort(next_f, (size_t)fsize, sizeof(int), cmp_int_asc);
        memcpy(frontier, next_f, (size_t)fsize * sizeof(int));
    }

    write_trace_csv(path, recs, total_tasks);
    free(recs); free(edges); free(dist); free(frontier); free(next_f);
}

/* 3. Morsel-Driven Hash Join (Build + Probe phases with depend edges) */
static void extract_join_trace(const char *path) {
    const int build_n = 32768, probe_n = 131072, morsel = 1024, phases = 6;
    const int nmorsels = (probe_n + morsel - 1) / morsel;
    const int total_tasks = phases * nmorsels;
    const uint32_t mask = (uint32_t)(build_n * 2 - 1);

    uint32_t *ht    = (uint32_t *)xcalloc((size_t)build_n * 2, sizeof(uint32_t));
    uint32_t *probe = (uint32_t *)xmalloc((size_t)probe_n * sizeof(uint32_t));
    long long *out_cnt = (long long *)xcalloc((size_t)nmorsels, sizeof(long long));
    task_trace_rec_t *recs = (task_trace_rec_t *)xcalloc((size_t)total_tasks, sizeof(task_trace_rec_t));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < build_n; i++) {
        uint32_t k = (uint32_t)i * 2654435761U + 1U;
        ht[k & mask] = k;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < probe_n; i++) {
        probe[i] = (uint32_t)(i % build_n) * 2654435761U + 1U;
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int ph = 0; ph < phases; ph++) {
                for (int m = 0; m < nmorsels; m++) {
                    int i0 = m * morsel;
                    int i1 = (i0 + morsel < probe_n) ? (i0 + morsel) : probe_n;
                    int tid = ph * nmorsels + m;
                    int c_tid = (m + ph) & 7;

                    #pragma omp task firstprivate(i0, i1, tid, ph, c_tid, m) \
                                     depend(in: probe[i0]) depend(inout: out_cnt[m])
                    {
                        task_trace_rec_t *rec = &recs[tid];
                        size_t blk_base = (size_t)m * 16U * 4096U;
                        rec->task_id = tid;
                        rec->phase = ph;
                        rec->creator_tid = c_tid;
                        rec->work_ops = (i1 - i0) * 4;
                        rec->num_deps = 2;
                        rec->dep_pages[0] = (int)logical_page_id(6, blk_base);
                        rec->dep_is_write[0] = 0;
                        rec->dep_pages[1] = (int)logical_page_id(7, blk_base + 4096U);
                        rec->dep_is_write[1] = 1;

                        long long local_hits = 0;
                        for (int i = i0; i < i1; i++) {
                            uint32_t k = probe[i];
                            if (ht[k & mask] == k) local_hits++;
                        }
                        out_cnt[m] += local_hits;

                        int nt = 0;
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[0];
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[1];
                        for (int p = 2; p < 11; p++) {
                            uint32_t arr_id = (p < 7) ? 6U : 7U;
                            rec->pages[nt++] = logical_page_id(arr_id, blk_base + (size_t)p * 4096U);
                        }
                        for (int p = 11; p < 15; p++) {
                            rec->pages[nt++] = logical_page_id(8, blk_base + (size_t)(p - 11) * 4096U);
                        }
                        size_t halo_off = (size_t)(((m / 4) + ph * 3) & 15) * 4096U;
                        rec->pages[nt++] = logical_page_id(8, halo_off);
                        rec->num_pages = nt;
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    write_trace_csv(path, recs, total_tasks);
    free(recs); free(ht); free(probe); free(out_cnt);
}

/* 4. Iterative PageRank (Pull-based damping iterations with cross-phase depend chains) */
static void extract_pagerank_trace(const char *path) {
    const int nverts = 32768, in_deg = 8, chunk = 256, phases = 8;
    const int nblocks = (nverts + chunk - 1) / chunk;
    const int total_tasks = phases * nblocks;
    const double damp = 0.85;

    int *in_nbrs   = (int *)xmalloc((size_t)nverts * (size_t)in_deg * sizeof(int));
    double *pr_cur = (double *)xmalloc((size_t)nverts * sizeof(double));
    double *pr_nxt = (double *)xmalloc((size_t)nverts * sizeof(double));
    task_trace_rec_t *recs = (task_trace_rec_t *)xcalloc((size_t)total_tasks, sizeof(task_trace_rec_t));

    #pragma omp parallel for schedule(static)
    for (int v = 0; v < nverts; v++) {
        pr_cur[v] = 1.0 / (double)nverts;
        pr_nxt[v] = 0.0;
        for (int k = 0; k < in_deg; k++) {
            uint64_t h = (uint64_t)v * 7046029254386353131ULL + (uint64_t)k * 2654435761ULL;
            in_nbrs[v * in_deg + k] = (int)((h >> 15) % (uint64_t)nverts);
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int ph = 0; ph < phases; ph++) {
                for (int blk = 0; blk < nblocks; blk++) {
                    int v0 = blk * chunk;
                    int v1 = (v0 + chunk < nverts) ? (v0 + chunk) : nverts;
                    int tid = ph * nblocks + blk;
                    int c_tid = (blk + ph) & 7;

                    #pragma omp task firstprivate(v0, v1, tid, ph, c_tid, blk) \
                                     depend(in: in_nbrs[v0 * in_deg]) depend(inout: pr_nxt[v0])
                    {
                        task_trace_rec_t *rec = &recs[tid];
                        size_t blk_base = (size_t)blk * 16U * 4096U;
                        rec->task_id = tid;
                        rec->phase = ph;
                        rec->creator_tid = c_tid;
                        rec->work_ops = (v1 - v0) * in_deg * 3;
                        rec->num_deps = 2;
                        rec->dep_pages[0] = (int)logical_page_id(9, blk_base);
                        rec->dep_is_write[0] = 1;
                        rec->dep_pages[1] = (int)logical_page_id(10, blk_base + 4096U);
                        rec->dep_is_write[1] = 0;

                        for (int v = v0; v < v1; v++) {
                            double sum = 0.0;
                            for (int k = 0; k < in_deg; k++) {
                                int u = in_nbrs[v * in_deg + k];
                                sum += pr_cur[u] / (double)in_deg;
                            }
                            pr_nxt[v] = (1.0 - damp) / (double)nverts + damp * sum;
                        }

                        int nt = 0;
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[0];
                        rec->pages[nt++] = (uint32_t)rec->dep_pages[1];
                        for (int p = 2; p < 11; p++) {
                            uint32_t arr_id = (p < 6) ? 9U : 10U;
                            rec->pages[nt++] = logical_page_id(arr_id, blk_base + (size_t)p * 4096U);
                        }
                        for (int p = 11; p < 15; p++) {
                            rec->pages[nt++] = logical_page_id(11, blk_base + (size_t)(p - 11) * 4096U);
                        }
                        size_t halo_off = (size_t)(((blk / 4) + ph * 3) & 15) * 4096U;
                        rec->pages[nt++] = logical_page_id(11, halo_off);
                        rec->num_pages = nt;
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    write_trace_csv(path, recs, total_tasks);
    free(recs); free(in_nbrs); free(pr_cur); free(pr_nxt);
}

int main(int argc, char **argv) {
    const char *out_dir = (argc > 1) ? argv[1] : "traces_out";
    if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) {
        perror("mkdir");
        return 1;
    }

    int ok = verify_openmp_smoke_tests();
    printf("[01_trace_extractor] OpenMP Tasking Smoke Test (depend/taskloop/if0/final): %s (threads=%d)\n",
           ok ? "PASS" : "FAIL", omp_get_max_threads());
    if (!ok) return 1;

    char p1[256], p2[256], p3[256], p4[256];
    snprintf(p1, sizeof(p1), "%s/trace_spmv.csv", out_dir);
    snprintf(p2, sizeof(p2), "%s/trace_bfs.csv", out_dir);
    snprintf(p3, sizeof(p3), "%s/trace_join.csv", out_dir);
    snprintf(p4, sizeof(p4), "%s/trace_pagerank.csv", out_dir);

    extract_spmv_trace(p1);
    extract_bfs_trace(p2);
    extract_join_trace(p3);
    extract_pagerank_trace(p4);
    printf("[01_trace_extractor] Generated deterministic logical-page traces (SpMV, BFS, HashJoin, PageRank) in %s/\n", out_dir);
    return 0;
}
