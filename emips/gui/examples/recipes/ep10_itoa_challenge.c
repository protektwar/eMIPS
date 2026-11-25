/* MIPS by Example in eMIPS, episode 10, the challenge solved (the C version
 * of ep10_itoa_challenge.asm). */
#include "emips.h"

static const int numbers[] = {2026, -42, -2147483647 - 1, 0};
static char buf[12];
static char hexbuf[9];

static char *utoa(unsigned n, unsigned b, char *end)
{
    *end = '\0';
    do {
        *--end = "0123456789abcdef"[n % b];
        n /= b;
    } while (n != 0);
    return end;
}

char *itoa(int n, char *end)
{
    unsigned u = n < 0 ? 0u - (unsigned)n : (unsigned)n;   /* wraps: no overflow */
    char *p = utoa(u, 10, end);
    if (n < 0)
        *--p = '-';
    return p;
}

void hex8(unsigned n, char *out)
{
    for (int shift = 28; shift >= 0; shift -= 4)
        *out++ = "0123456789abcdef"[(n >> shift) & 0xf];
    *out = '\0';
}

int main(void)
{
    for (int i = 0; i < 4; i++) {
        print_string(itoa(numbers[i], buf + 11));
        print_string(" = 0x");
        hex8(numbers[i], hexbuf);
        print_string(hexbuf);
        print_char('\n');
    }
    return 0;
}
