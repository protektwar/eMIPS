/* Lab: brighten a 64-pixel image in C (python3 -m emips cc c_dsp.c -O 2, and
 * cc c_dsp.c -O 2 --dspr2; prebuilt: c_dsp_plain.elf, c_dsp.elf).  With the
 * DSP Module (__mips_dsp: -mdsp / -mdspr2) brighten() uses GCC's builtins on
 * four pixels at a time; without it, a pixel at a time.  main prints the
 * sum of the pixels and the last one. */
#include "emips.h"

#define N 64
static unsigned char image[N] __attribute__((aligned(4)));

#ifdef __mips_dsp
typedef signed char v4i8 __attribute__((vector_size(4)));

static void brighten(unsigned char *p, int n, int add)
{
    v4i8 *q = (v4i8 *)p;
    v4i8 k = __builtin_mips_repl_qb(add);           /* add in all four bytes */
    for (int i = 0; i < n / 4; i++)
        q[i] = __builtin_mips_addu_s_qb(q[i], k);    /* saturating at 255 */
}
#else
static void brighten(unsigned char *p, int n, int add)
{
    for (int i = 0; i < n; i++) {
        int v = p[i] + add;
        p[i] = v > 255 ? 255 : v;
    }
}
#endif

int main(void)
{
    for (int i = 0; i < N; i++)
        image[i] = i * 4;                            /* 0, 4, ... 252 */
    brighten(image, N, 0x40);
    unsigned sum = 0;
    for (int i = 0; i < N; i++)
        sum += image[i];
    print_unsigned(sum);
    print_char(' ');
    print_unsigned(image[N - 1]);
    print_char('\n');
    return 0;
}
