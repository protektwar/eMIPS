/* MIPS by Example in eMIPS, episode 47 — Two cores and a lock (the C version
 * of ep47_lock.asm).  Both cores run main.  The lock is GCC's atomic
 * exchange — ll / sc underneath — and its release a store with a barrier.
 * The shared words live in .data: crt0 clears .bss on each core as it
 * starts, and core 1 would wipe what core 0 had already counted.  Core 1
 * never returns from main: that would end the program. */
#include "emips.h"

#define N 1000                          /* additions per core */
#define SHARED __attribute__((section(".data")))

static volatile int lock SHARED, count SHARED, done SHARED;

static void acquire(volatile int *l)
{
    while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE))
        ;                               /* taken: try again */
}

static void release(volatile int *l)
{
    __atomic_store_n(l, 0, __ATOMIC_RELEASE);
}

int main(void)
{
    int core = MFC0(15, 1) & 0x3ff;     /* EBase.CPUNum: which core am I? */
    for (int i = 0; i < N; i++) {
        acquire(&lock);
        count = count + 1;              /* one core at a time */
        release(&lock);
    }
    if (core != 0) {                    /* core 1: say so, and stop */
        __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
        for (;;)
            __asm__ volatile ("wait");
    }
    while (!done)                       /* core 0: wait for core 1, then print */
        ;
    print_string("count = ");
    print_int(count);
    print_string(" (2 cores x 1000)\n");
    return 0;
}
// eMIPS: --cores 2
