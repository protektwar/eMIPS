/* MIPS by Example in eMIPS, episode 39 — An interrupt-driven keyboard (the C
 * version of ep39_keyboard.asm).  The handler (EMIPS_EXCEPTION_HANDLER: an
 * assembly entry at 0x80000180 that calls on_interrupt) puts each key in a
 * ring buffer; main sleeps with WAIT until there is one. */
#include "emips.h"

static volatile unsigned char buf[16];  /* the ring buffer */
static volatile unsigned head, tail;    /* the handler writes at head, main reads at tail */

unsigned on_interrupt(unsigned cause, unsigned epc, unsigned badvaddr)
{
    (void)badvaddr;
    if (cause & 0x0800) {               /* IP3: the keyboard */
        buf[head] = MMIO_RX_DATA;       /* the key (its Ready bit clears) */
        head = (head + 1) & 15;
    }
    return epc;                         /* an interrupt: back to where it was */
}
EMIPS_EXCEPTION_HANDLER(on_interrupt);

static int upper(int c)
{
    return c >= 'a' && c <= 'z' ? c - 32 : c;
}

int main(void)
{
    int keys = 0;
    print_string("type, Enter ends: ");
    MMIO_RX_CONTROL = 2;                /* interrupt when a key is ready */
    MTC0(12, 0, MFC0(12, 0) | 0x0801);  /* Status: IM3 and IE */
    for (;;) {
        while (tail == head)
            __asm__ volatile ("wait");  /* nothing yet: sleep until an interrupt */
        int c = buf[tail];
        tail = (tail + 1) & 15;
        if (c == '\n')
            break;
        keys++;
        print_char(upper(c));
    }
    MTC0(12, 0, MFC0(12, 0) & ~0x0801u);   /* interrupts off again */
    print_char('\n');
    print_string("keys: ");
    print_int(keys);
    print_char('\n');
    return 0;
}
// eMIPS: --key-delay 2000
