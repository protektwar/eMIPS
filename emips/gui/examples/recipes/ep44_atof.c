/* MIPS by Example in eMIPS, episode 44 — From text to a float (the C version
 * of ep44_atof.asm): the digits as an integer, the digits after the point
 * counted, then one division by 10^n — the only rounding. */
#include "emips.h"

static const char *inputs[] = {"3.14159", "-0.5", "100", "2.75", "0.1", 0};

static float parse_float(const char *s)
{
    int m = 0, after = 0, point = 0, minus = 0;
    if (*s == '-') {
        minus = 1;
        s++;
    }
    for (;; s++) {
        if (*s == '.')
            point = 1;
        else if (*s >= '0' && *s <= '9') {
            m = m * 10 + (*s - '0');
            after += point;             /* one more after the point (if past it) */
        } else
            break;
    }
    float p = 1.0f;                     /* 10 ^ after: exact up to 10^10 */
    while (after--)
        p *= 10.0f;
    float f = (float)m / p;             /* one rounding: the nearest float */
    return minus ? -f : f;
}

int main(void)
{
    for (const char **t = inputs; *t; t++) {
        float f = parse_float(*t);
        union { float f; unsigned u; } bits = {f};
        print_string(*t);
        print_string(" -> ");
        print_float(f);
        print_string(" = ");
        print_hex(bits.u);
        print_char('\n');
    }
    return 0;
}
