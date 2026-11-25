/* MIPS by Example in eMIPS, episode 10 — Numbers to text: decimal, binary,
 * hex (the C version of ep10_itoa.asm). */
#include "emips.h"

static const int numbers[] = {2026, -42};
static char buf[34];

/* the digits of n in base b, written from the end of buf; the first one */
char *utoa(unsigned n, unsigned b, char *end)
{
    *end = '\0';
    do {
        *--end = "0123456789abcdef"[n % b];
        n /= b;
    } while (n != 0);
    return end;
}

int main(void)
{
    for (int i = 0; i < 2; i++) {
        int n = numbers[i];
        if (n < 0)
            print_char('-');
        print_string(utoa(n < 0 ? -n : n, 10, buf + 33));
        print_string(" = 0b");
        print_string(utoa(n, 2, buf + 33));
        print_string(" = 0x");
        print_string(utoa(n, 16, buf + 33));
        print_char('\n');
    }
    return 0;
}
