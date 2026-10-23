/* Lab 72: what GCC's auto-vectorizer makes of a C loop (python3 -m emips cc
 * c_msa.c -O 3, and cc c_msa.c -O 3 --msa; prebuilt: c_msa_plain.elf,
 * c_msa.elf).  The source is the same for both: no builtins, no vector
 * types — with -mmsa GCC uses MSA by itself. */
#include "emips.h"

#define N 256

static int a[N], b[N], c[N];

int main(void)
{
    for (int i = 0; i < N; i++) {          /* the data */
        a[i] = i;
        b[i] = N - i;
    }
    for (int i = 0; i < N; i++)            /* the loop to vectorize */
        c[i] = 3 * a[i] + b[i];
    int sum = 0;
    for (int i = 0; i < N; i++)            /* a reduction */
        sum += c[i];
    print_int(sum);
    print_char('\n');
    return 0;
}
