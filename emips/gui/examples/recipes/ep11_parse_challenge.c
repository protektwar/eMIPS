/* MIPS by Example in eMIPS, episode 11, the challenge solved (the C version
 * of ep11_parse_challenge.asm). */
#include "emips.h"

static char buf[64];

static int digit(int c, int base)        /* its value, or -1 */
{
    int v = c - '0';
    if ((unsigned)v >= 10) {
        v = (c | 0x20) - 'a' + 10;
        if (v < 10 || v >= 16)
            return -1;
    }
    return v < base ? v : -1;
}

const char *parse_num(const char *s, int *n)
{
    int negative = 0, base = 10, v = 0, d;

    while (*s == ' ')
        s++;
    if (*s == '-') {
        negative = 1;
        s++;
    }
    if (s[0] == '0' && s[1] == 'x') {
        base = 16;
        s += 2;
    }
    if (digit((unsigned char)*s, base) < 0)
        return 0;
    for (; (d = digit((unsigned char)*s, base)) >= 0; s++)
        v = v * base + d;
    *n = negative ? -v : v;
    return s;
}

int main(void)
{
    const char *p = buf;
    int n, count = 0, sum = 0;

    print_string("Numbers? ");
    read_string(buf, 64);
    while ((p = parse_num(p, &n)) != 0) {
        count++;
        sum += n;
    }
    print_string("count ");
    print_int(count);
    print_string(", sum ");
    print_int(sum);
    print_string(", average ");
    print_int(sum / count);
    print_string(" remainder ");
    print_int(sum % count);
    print_char('\n');
    return 0;
}
