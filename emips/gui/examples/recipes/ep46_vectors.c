/* MIPS by Example in eMIPS, episode 46 — Vectors: MSA from assembly and C
 * (the C version of ep46_vectors.asm).  msa.h gives the 128-bit types (v4i32:
 * four ints) and a builtin for each MSA instruction; eMIPS builds a program
 * that includes it with -mmsa, for Release 5. */
#include "emips.h"
#include <msa.h>

#define N 8                             /* a multiple of 4 */

static int a[N] __attribute__((aligned(16))) = {1, 2, 3, 4, 5, 6, 7, 8};
static int b[N] __attribute__((aligned(16))) = {10, 20, 30, 40, 50, 60, 70, 80};
static int c[N] __attribute__((aligned(16)));

/* four sums in the lanes, then the lanes added */
static int sum(const int *v, int n)
{
    v4i32 s = __builtin_msa_ldi_w(0);
    for (int i = 0; i < n; i += 4)
        s = __builtin_msa_addv_w(s, __builtin_msa_ld_w((void *)(v + i), 0));
    return __builtin_msa_copy_s_w(s, 0) + __builtin_msa_copy_s_w(s, 1)
         + __builtin_msa_copy_s_w(s, 2) + __builtin_msa_copy_s_w(s, 3);
}

static void print_array(const int *v, int n)
{
    for (int i = 0; i < n; i++) {
        print_int(v[i]);
        print_char(i < n - 1 ? ' ' : '\n');
    }
}

int main(void)
{
    for (int i = 0; i < N; i += 4) {    /* c = a + b, four words a step */
        v4i32 x = __builtin_msa_ld_w(a + i, 0);
        v4i32 y = __builtin_msa_ld_w(b + i, 0);
        __builtin_msa_st_w(__builtin_msa_addv_w(x, y), c + i, 0);
    }
    print_string("c = a + b: ");
    print_array(c, N);
    print_string("sum of c: ");
    print_int(sum(c, N));
    print_char('\n');
    return 0;
}
// eMIPS: --release R5
