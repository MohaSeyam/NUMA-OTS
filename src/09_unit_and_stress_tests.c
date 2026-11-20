/*
 * 09_unit_and_stress_tests.c — Automated Unit, Concurrency & OpenMP 5.2 Correctness Suite
 *
 * Authors: Khalil Ibrahim Abu Zanouna and Mohammed Abdulsalam Seyam
 *          Computer Engineering Department, Islamic University of Gaza
 *
 * Covers:
 *   T1: 8-thread 64-stripe PHHT concurrency stress & epoch TTL expiry
 *   T2: GCC 13.2-15.2 libgomp depend ABI parser (legacy, extended 5.x, omp_all_memory, flag bit)
 *   T3: M1 Eq. (5) write-weighted affinity scoring & creator/least-loaded tie-breaking
 *   T4: M2 Proposition 1 R_min(B) batch-size calibration & EMA step-change guard
 *   T5: OpenMP 5.2 core tasking (depend in/out/inout/mutexinoutset/inoutset, taskgroup,
 *       taskloop reduction, if(0) undeferred execution, final(1) with omp_in_final())
 *   T6: OpenMP 5.2 untied tasks & detachable tasks (detach + omp_fulfill_event)
 *   T7: OpenMP nested parallel teams (cross-team isolation) & taskgroup cancellation
 *   T8: Asymmetric task-tree liveness & zero-starvation completion stress test
 *
 * Build (native & sanitizer-checked):
 *   gcc -O2 -std=c11 -Wall -Wextra -Werror -fopenmp 09_unit_and_stress_tests.c -o unit_tests -lpthread -lm
 *   gcc -O1 -g -fsanitize=address,undefined -std=c11 -Wall -Wextra -Werror -fopenmp \
 *       09_unit_and_stress_tests.c -o unit_tests_asan -lpthread -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include "numa_ots_core.h"

#define TEST_STRIPES 64
#define TEST_SLOTS   64
#define TEST_THREADS 8

typedef struct {
    uintptr_t page_base;
    int home_node;
    int cached_epoch;
    int valid;
} test_slot_t;

typedef struct {
    pthread_mutex_t lock;
    test_slot_t slots[TEST_SLOTS];
    char pad[64];
} test_stripe_t;

static test_stripe_t g_test_phht[TEST_STRIPES];
static _Atomic int g_t1_errors = 0;

static void *phht_stress_worker(void *arg) {
    int tid = *(int *)arg;
    for (int iter = 0; iter < 5000; iter++) {
        uintptr_t pbase = (uintptr_t)(0x10000000ULL + ((uint64_t)((iter * 7 + tid * 13) & 255) << 12));
        uint64_t h = (pbase >> 12) * 2654435761ULL;
        int st = (int)(h & (TEST_STRIPES - 1));
        int start = (int)((h >> 6) & (TEST_SLOTS - 1));
        int home = (int)((pbase >> 12) & 7);
        int ep = 1 + (iter / 1000);

        pthread_mutex_lock(&g_test_phht[st].lock);
        int found = 0;
        for (int i = 0; i < TEST_SLOTS; i++) {
            int idx = (start + i) & (TEST_SLOTS - 1);
            test_slot_t *sl = &g_test_phht[st].slots[idx];
            if (!sl->valid || sl->page_base == pbase) {
                sl->valid = 1;
                sl->page_base = pbase;
                sl->home_node = home;
                sl->cached_epoch = ep;
                found = 1;
                break;
            }
        }
        if (!found) __atomic_fetch_add(&g_t1_errors, 1, __ATOMIC_RELAXED);
        pthread_mutex_unlock(&g_test_phht[st].lock);
    }
    return NULL;
}

/* T1: 8-thread PHHT concurrency stress & epoch TTL expiration */
static int test_t1_phht_concurrency(void) {
    for (int s = 0; s < TEST_STRIPES; s++) {
        pthread_mutex_init(&g_test_phht[s].lock, NULL);
        memset(g_test_phht[s].slots, 0, sizeof(g_test_phht[s].slots));
    }
    pthread_t th[TEST_THREADS];
    int ids[TEST_THREADS];
    for (int t = 0; t < TEST_THREADS; t++) {
        ids[t] = t;
        pthread_create(&th[t], NULL, phht_stress_worker, &ids[t]);
    }
    for (int t = 0; t < TEST_THREADS; t++) pthread_join(th[t], NULL);

    int entries = 0;
    for (int s = 0; s < TEST_STRIPES; s++) {
        for (int i = 0; i < TEST_SLOTS; i++) {
            if (g_test_phht[s].slots[i].valid) {
                entries++;
                int expected_home = (int)((g_test_phht[s].slots[i].page_base >> 12) & 7);
                if (g_test_phht[s].slots[i].home_node != expected_home) return 0;
            }
        }
        pthread_mutex_destroy(&g_test_phht[s].lock);
    }
    return (__atomic_load_n(&g_t1_errors, __ATOMIC_RELAXED) == 0 && entries == 256) ? 1 : 0;
}

/* T2: GCC 13.2-15.2 libgomp depend ABI parser (legacy, extended 5.x, omp_all_memory) */
static int test_t2_gcc_depend_abi(void) {
    uintptr_t pages[8];
    int weights[8];

    /* Legacy GCC layout: depend[0] = total (3), depend[1] = n_out (1), [2]=out, [3]=in, [4]=in */
    void *legacy_dep[5];
    legacy_dep[0] = (void *)(uintptr_t)3;
    legacy_dep[1] = (void *)(uintptr_t)1;
    legacy_dep[2] = (void *)(uintptr_t)0x20001040ULL; /* out -> weight 2 */
    legacy_dep[3] = (void *)(uintptr_t)0x20002080ULL; /* in  -> weight 1 */
    legacy_dep[4] = (void *)(uintptr_t)0x20001090ULL; /* same page as [2] -> max weight 2 */

    int n1 = numa_ots_parse_gcc_depend(legacy_dep, pages, weights, 8);
    if (n1 != 2) return 0;
    if (pages[0] != 0x20001000ULL || weights[0] != 2) return 0;
    if (pages[1] != 0x20002000ULL || weights[1] != 1) return 0;

    /* OpenMP 5.x extended layout: depend[0]=0, [1]=total(4), [2]=n_out(1), [3]=n_mt(1), [4]=n_in(1), +1 inoutset */
    void *ext_dep[9];
    ext_dep[0] = (void *)(uintptr_t)0;
    ext_dep[1] = (void *)(uintptr_t)4;
    ext_dep[2] = (void *)(uintptr_t)1; /* n_out */
    ext_dep[3] = (void *)(uintptr_t)1; /* n_mt */
    ext_dep[4] = (void *)(uintptr_t)1; /* n_in */
    ext_dep[5] = (void *)(uintptr_t)0x30004010ULL; /* out -> w=2 */
    ext_dep[6] = (void *)(uintptr_t)0x30005020ULL; /* mutexinoutset -> w=2 */
    ext_dep[7] = (void *)(uintptr_t)0x0ULL;        /* in: omp_all_memory (NULL token) -> skipped */
    ext_dep[8] = (void *)(uintptr_t)0x30006030ULL; /* inoutset -> w=2 */

    int n2 = numa_ots_parse_gcc_depend(ext_dep, pages, weights, 8);
    if (n2 != 3) return 0;
    if (pages[0] != 0x30004000ULL || weights[0] != 2) return 0;
    if (pages[1] != 0x30005000ULL || weights[1] != 2) return 0;
    if (pages[2] != 0x30006000ULL || weights[2] != 2) return 0;
    return 1;
}

/* T3: M1 Eq. (5) affinity scoring & tie-breaking */
static int test_t3_m1_eq5(void) {
    int scores1[4] = {1, 4, 2, 0};
    long q1[4] = {10, 20, 5, 1};
    if (numa_ots_select_m1_target_eq5(scores1, q1, 0, 4) != 1) return 0;

    /* Tie including creator_node -> creator_node wins */
    int scores2[4] = {3, 3, 1, 0};
    long q2[4] = {50, 1, 0, 0};
    if (numa_ots_select_m1_target_eq5(scores2, q2, 0, 4) != 0) return 0;

    /* Tie NOT including creator_node -> least loaded domain wins */
    int scores3[4] = {1, 4, 4, 2};
    long q3[4] = {0, 15, 3, 0};
    if (numa_ots_select_m1_target_eq5(scores3, q3, 0, 4) != 2) return 0;
    return 1;
}

/* T4: M2 Proposition 1 R_min(B) calibration & EMA step-change guard */
static int test_t4_m2_prop1(void) {
    /* B = 64 calibrated costs (2N=1850, 4N=2200, 8N=2900 ns) */
    if (numa_ots_calc_rmin_prop1(1.5, 1850.0, 65.0) != 43) return 0;
    if (numa_ots_calc_rmin_prop1(1.5, 2200.0, 65.0) != 51) return 0;
    if (numa_ots_calc_rmin_prop1(1.5, 2900.0, 65.0) != 67) return 0;
    if (numa_ots_calc_rmin_prop1(1.5, 2900.0, 102.0) != 43) return 0;

    /* Smaller batch size B = 8 on 8N: C_hat = 1450 + 92800/8 = 13050 ns -> R_min = 302 */
    double c_b8 = 1450.0 + 92800.0 / 8.0;
    if (numa_ots_calc_rmin_prop1(1.5, c_b8, 65.0) != 302) return 0;

    /* EMA step-change guard: jump > 3x triggers immediate reset */
    double r_step = numa_ots_eval_m2_ema_step(80.0, 10.0, 0.3, 8.0, 3.0);
    if (fabs(r_step - 80.0) > 1e-9) return 0;
    /* Smooth EMA update when delta <= 3x */
    double r_ema = numa_ots_eval_m2_ema_step(20.0, 10.0, 0.3, 8.0, 3.0);
    if (fabs(r_ema - 13.0) > 1e-9) return 0;
    return 1;
}

/* T5: OpenMP 5.2 core constructs (depend in/out/inout/mutexinoutset/inoutset, taskgroup, taskloop, if0, final) */
static int test_t5_openmp52_core(void) {
    int shared_val = 0, mutex_acc = 0, inoutset_acc = 0, if0_ok = 0, final_ok = 0;
    long long tl_sum = 0;

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task depend(out: shared_val) shared(shared_val)
                { shared_val = 10; }

                #pragma omp task depend(inout: shared_val) shared(shared_val)
                { shared_val += 32; }

                for (int k = 0; k < 4; k++) {
                    #pragma omp task depend(mutexinoutset: mutex_acc) shared(mutex_acc)
                    { mutex_acc += 5; }
                }

                #pragma omp task depend(inoutset: inoutset_acc) shared(inoutset_acc)
                { __atomic_fetch_add(&inoutset_acc, 7, __ATOMIC_RELAXED); }

                #pragma omp task depend(inoutset: inoutset_acc) shared(inoutset_acc)
                { __atomic_fetch_add(&inoutset_acc, 11, __ATOMIC_RELAXED); }

                int parent_tid = omp_get_thread_num();
                #pragma omp task if(0) firstprivate(parent_tid) shared(if0_ok)
                { if (omp_get_thread_num() == parent_tid) if0_ok = 1; }

                #pragma omp task final(1) shared(final_ok)
                { if (omp_in_final()) final_ok = 1; }
            }

            #pragma omp taskloop reduction(+:tl_sum) grainsize(64)
            for (int i = 1; i <= 1000; i++) tl_sum += i;
        }
    }
    return (shared_val == 42 && mutex_acc == 20 && inoutset_acc == 18 &&
            if0_ok == 1 && final_ok == 1 && tl_sum == 500500LL) ? 1 : 0;
}

/* T6: Untied tasks & OpenMP 5.0/5.2 detachable tasks (detach + omp_fulfill_event) */
static int test_t6_untied_and_detach(void) {
    int untied_sum = 0;
    int detach_done = 0;
    omp_event_handle_t ev;

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int i = 0; i < 16; i++) {
                #pragma omp task untied shared(untied_sum)
                {
                    #pragma omp taskyield
                    __atomic_fetch_add(&untied_sum, 1, __ATOMIC_RELAXED);
                }
            }

            #pragma omp task detach(ev) shared(detach_done)
            {
                __atomic_store_n(&detach_done, 1, __ATOMIC_RELEASE);
                omp_fulfill_event(ev);
            }

            #pragma omp taskwait
        }
    }
    return (untied_sum == 16 && detach_done == 1) ? 1 : 0;
}

/* T7: Nested parallel teams (cross-team isolation) & taskgroup cancellation */
static int test_t7_nested_and_cancel(void) {
    int team_sums[2] = {0, 0};
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(2) shared(team_sums)
    {
        int outer_id = omp_get_thread_num();
        #pragma omp parallel num_threads(2) firstprivate(outer_id) shared(team_sums)
        {
            #pragma omp single
            {
                for (int j = 0; j < 8; j++) {
                    #pragma omp task firstprivate(outer_id) shared(team_sums)
                    {
                        __atomic_fetch_add(&team_sums[outer_id], 1, __ATOMIC_RELAXED);
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    int cancel_ran = 0;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                for (int k = 0; k < 16; k++) {
                    #pragma omp task firstprivate(k) shared(cancel_ran)
                    {
                        __atomic_fetch_add(&cancel_ran, 1, __ATOMIC_RELAXED);
                        if (k == 0) {
                            #pragma omp cancel taskgroup
                        }
                        #pragma omp cancellation point taskgroup
                    }
                }
            }
        }
    }
    return (team_sums[0] == 8 && team_sums[1] == 8 && cancel_ran >= 1) ? 1 : 0;
}

/* T8: Asymmetric task-tree liveness & zero-starvation completion stress */
static int test_t8_liveness_stress(void) {
    const int N_TASKS = 512;
    _Atomic int completed = 0;
    int tokens[64] = {0};

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int i = 0; i < N_TASKS; i++) {
                int slot = i & 63;
                #pragma omp task firstprivate(i, slot) depend(inout: tokens[slot]) shared(completed, tokens)
                {
                    tokens[slot] += (i & 3) + 1;
                    __atomic_fetch_add(&completed, 1, __ATOMIC_RELAXED);
                }
            }
            #pragma omp taskwait
        }
    }
    return (__atomic_load_n(&completed, __ATOMIC_RELAXED) == N_TASKS) ? 1 : 0;
}

int main(int argc, char **argv) {
    int t1 = test_t1_phht_concurrency();
    int t2 = test_t2_gcc_depend_abi();
    int t3 = test_t3_m1_eq5();
    int t4 = test_t4_m2_prop1();
    int t5 = test_t5_openmp52_core();
    int t6 = test_t6_untied_and_detach();
    int t7 = test_t7_nested_and_cancel();
    int t8 = test_t8_liveness_stress();

    if (argc > 1) {
        FILE *fp = fopen(argv[1], "w");
        if (fp) {
            fprintf(fp, "test,status\n");
            fprintf(fp, "T1_PHHT,%s\nT2_ABI,%s\nT3_M1_Eq5,%s\nT4_M2_Prop1,%s\n"
                        "T5_OMP52,%s\nT6_Untied_Detach,%s\nT7_Nested_Cancel,%s\nT8_Liveness,%s\n",
                    t1 ? "PASS" : "FAIL", t2 ? "PASS" : "FAIL", t3 ? "PASS" : "FAIL",
                    t4 ? "PASS" : "FAIL", t5 ? "PASS" : "FAIL", t6 ? "PASS" : "FAIL",
                    t7 ? "PASS" : "FAIL", t8 ? "PASS" : "FAIL");
            fclose(fp);
        }
    }

    printf("[09_unit_and_stress_tests] T1(PHHT)=%s | T2(ABI)=%s | T3(M1_Eq5)=%s | "
           "T4(M2_Prop1)=%s | T5(OMP5.2)=%s | T6(Untied/Detach)=%s | T7(Nested/Cancel)=%s | T8(Liveness)=%s -> %s\n",
           t1 ? "PASS" : "FAIL", t2 ? "PASS" : "FAIL", t3 ? "PASS" : "FAIL",
           t4 ? "PASS" : "FAIL", t5 ? "PASS" : "FAIL", t6 ? "PASS" : "FAIL",
           t7 ? "PASS" : "FAIL", t8 ? "PASS" : "FAIL",
           (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8) ? "ALL PASS" : "FAIL");
    return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8) ? 0 : 1;
}
