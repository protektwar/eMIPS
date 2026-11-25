/* MIPS by Example in eMIPS, episode 11 — Text to numbers: parse a line, min
 * and max (the C version of ep11_parse.asm). */
#include "emips.h"

static char buf[64];

/* the number at the start of s (spaces skipped) in *n; where it stopped,
 * or 0 if there is no number */
const char *parse_int(const char *s, int *n)
{
    int negative = 0, v = 0;

    while (*s == ' ')
        s++;
    if (*s == '-') {
        negative = 1;
        s++;
    }
    if ((unsigned)(*s - '0') >= 10)
        return 0;
    for (; (unsigned)(*s - '0') < 10; s++)
        v = v * 10 + (*s - '0');
    *n = negative ? -v : v;
    return s;
}

int main(void)
{
    const char *p = buf;
    int n, count = 0, min = 0x7fffffff, max = -0x7fffffff - 1, sum = 0;

    print_string("Numbers? ");
    read_string(buf, 64);
    while ((p = parse_int(p, &n)) != 0) {
        count++;
        sum += n;
        if (n < min)
            min = n;
        if (n > max)
            max = n;
    }
    print_string("count ");
    print_int(count);
    print_string(", min ");
    print_int(min);
    print_string(", max ");
    print_int(max);
    print_string(", sum ");
    print_int(sum);
    print_char('\n');
    return 0;
}
