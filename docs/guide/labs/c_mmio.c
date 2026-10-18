/* Lab: volatile and memory-mapped I/O (python3 -m emips cc c_mmio.c -O 2).
 * Echo what is typed on the MMIO keyboard to the display, in upper case,
 * until a '.' (run with --keys "hi.").  The device registers are
 * volatile: every read must reach the device. */
#include "emips.h"

static char getc_mmio(void)
{
    while (!(MMIO_RX_CONTROL & 1))          /* wait for a key */
        ;
    return (char)MMIO_RX_DATA;
}

static void putc_mmio(char c)
{
    while (!(MMIO_TX_CONTROL & 1))          /* wait until the display is ready */
        ;
    MMIO_TX_DATA = (unsigned)c;
}

int main(void)
{
    for (;;) {
        char c = getc_mmio();
        if (c == '.')
            break;
        if (c >= 'a' && c <= 'z')
            c -= 'a' - 'A';
        putc_mmio(c);
    }
    putc_mmio('\n');
    return 0;
}
