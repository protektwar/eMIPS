/* MIPS by Example in eMIPS, episode 40 — the challenge, solved: a countdown
 * (the C version of ep40_clock_challenge.asm): 00:05 down to 00:00, then
 * "time's up!"; the handler turns its own interrupt off at zero.  Count counts up by itself; at Compare the
 * timer interrupt (Cause.IP7) runs on_interrupt, which counts a second and
 * moves Compare one period on.  This clock runs 100 times fast: a "second"
 * is 10000 cycles. */
#include "emips.h"

#define PERIOD  5000                    /* Count steps a "second" */
#define SECONDS 5

static volatile int seconds = SECONDS;  /* the handler counts down, main reads */

unsigned on_interrupt(unsigned cause, unsigned epc, unsigned badvaddr)
{
    (void)badvaddr;
    if (cause & 0x8000) {               /* IP7: the timer */
        if (--seconds == 0) {           /* zero: the timer's interrupt off */
            MTC0(12, 0, MFC0(12, 0) & ~0x8000u);
            MTC0(11, 0, 0);             /* (its pending interrupt cleared) */
        } else
            MTC0(11, 0, MFC0(11, 0) + PERIOD);
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
    int shown = SECONDS;
    print_string("the countdown starts\n");
    print_time(SECONDS);
    MTC0(11, 0, MFC0(9, 0) + PERIOD);   /* Compare = Count + one period */
    MTC0(12, 0, MFC0(12, 0) | 0x8001);  /* Status: IM7 and IE */
    while (shown > 0) {
        while (seconds == shown)
            __asm__ volatile ("wait");  /* sleep until an interrupt */
        shown = seconds;
        print_time(shown);
    }
    print_string("time's up!\n");
    return 0;
}
