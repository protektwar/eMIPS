/* MIPS by Example in eMIPS, episode 47 — the challenge, solved: no lock at
 * all (the C version of ep47_lock_challenge.asm): an atomic add, ll / addu /
 * sc in inline assembly, tried again when sc fails — and those failures
 * counted.  The shared words live in .data (crt0 clears .bss on each core). */
#include "emips.h"

#define N 1000                          /* additions per core */
#define SHARED __attribute__((section(".data")))

static volatile int count SHARED, done SHARED, fails1 SHARED;

/* *p += n as one indivisible step; returns how many times sc failed */
static int atomic_add(volatile int *p, int n)
{
    int fails = 0, t;
    for (;;) {
        __asm__ volatile ("ll    %0, 0(%1)\n\t"
                          "addu  %0, %0, %2\n\t"
                          "sc    %0, 0(%1)" : "=&r"(t) : "r"(p), "r"(n) : "memory");
        if (t)                          /* stored: nobody wrote it since the ll */
            return fails;
        fails++;
    }
}

int main(void)
{
    int core = MFC0(15, 1) & 0x3ff;     /* EBase.CPUNum: which core am I? */
    int fails = 0;
    for (int i = 0; i < N; i++)
        fails += atomic_add(&count, 1);
    if (core != 0) {                    /* core 1: its failures, done, and stop */
        fails1 = fails;
        __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
        for (;;)
            __asm__ volatile ("wait");
    }
    while (!done)                       /* core 0: wait for core 1, then print */
        ;
    print_string("count = ");
    print_int(count);
    print_string(" (2 cores x 1000)\n");
    print_string("failed sc: core 0 ");
    print_int(fails);
    print_string(", core 1 ");
    print_int(fails1);
    print_char('\n');
    return 0;
}
// eMIPS: --cores 2
