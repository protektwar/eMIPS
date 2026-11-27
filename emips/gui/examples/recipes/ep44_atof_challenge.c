/* MIPS by Example in eMIPS, episode 44 — the challenge, solved: an exponent
 * (the C version of ep44_atof_challenge.asm): the scale is the digits after
 * the point minus the exponent; a negative scale multiplies. */
#include "emips.h"

static const char *inputs[] = {"1.5e3", "2.5e-2", "-6.25E1", "1e10", "3.0e-5", 0};

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
    int e = 0, eminus = 0;
    if ((*s | 0x20) == 'e') {           /* E or e */
        s++;
        if (*s == '-') {
            eminus = 1;
            s++;
        }
        while (*s >= '0' && *s <= '9')
            e = e * 10 + (*s++ - '0');
    }
    int scale = after - (eminus ? -e : e);
    float p = 1.0f;                     /* 10 ^ |scale|: exact up to 10^10 */
    for (int k = scale < 0 ? -scale : scale; k > 0; k--)
        p *= 10.0f;
    float f = scale < 0 ? (float)m * p : (float)m / p;   /* one rounding */
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
