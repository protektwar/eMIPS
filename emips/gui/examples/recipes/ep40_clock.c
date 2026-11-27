/* MIPS by Example in eMIPS, episode 40 — A timer interrupt: a clock (the C
 * version of ep40_clock.asm).  Count counts up by itself; at Compare the
 * timer interrupt (Cause.IP7) runs on_interrupt, which counts a second and
 * moves Compare one period on.  This clock runs 100 times fast: a "second"
 * is 10000 cycles. */
#include "emips.h"

#define PERIOD  5000                    /* Count steps a "second" */
#define SECONDS 5

static volatile int seconds;            /* the handler counts, main reads */

unsigned on_interrupt(unsigned cause, unsigned epc, unsigned badvaddr)
{
    (void)badvaddr;
    if (cause & 0x8000) {               /* IP7: the timer */
        seconds++;
        MTC0(11, 0, MFC0(11, 0) + PERIOD);   /* Compare: the next second, no drift */
    }
    return epc;                         /* an interrupt: back to where it was */
}
EMIPS_EXCEPTION_HANDLER(on_interrupt);

static void two_digits(int n)
{
    print_char('0' + n / 10);
    print_char('0' + n % 10);
}

static void print_time(int s)
{
    two_digits(s / 60);
    print_char(':');
    two_digits(s % 60);
    print_char('\n');
}

int main(void)
{
    int shown = 0;
    print_string("the clock starts\n");
    MTC0(11, 0, MFC0(9, 0) + PERIOD);   /* Compare = Count + one period */
    MTC0(12, 0, MFC0(12, 0) | 0x8001);  /* Status: IM7 and IE */
    while (shown < SECONDS) {
        while (seconds == shown)
            __asm__ volatile ("wait");  /* sleep until an interrupt */
        shown = seconds;
        print_time(shown);
    }
    MTC0(12, 0, MFC0(12, 0) & ~0x8001u);   /* the timer's interrupt off again */
    print_string("the clock stops\n");
    return 0;
}
