/* MIPS by Example in eMIPS, episode 18, the challenge solved (the C version
 * of ep18_primes_challenge.asm). */
#include "emips.h"

static const unsigned fracs[6] = {84, 126, 1071, 462, 13, 17};

unsigned gcd(unsigned a, unsigned b)      /* recursively */
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

static void print_frac(unsigned n, unsigned d)
{
    print_int(n);
    print_string("/");
    print_int(d);
}

void reduce(unsigned n, unsigned d)
{
    unsigned g;

    print_frac(n, d);
    print_string(" = ");
    g = gcd(n, d);
    print_frac(n / g, d / g);
    print_char('\n');
}

int main(void)
{
    for (int i = 0; i < 6; i += 2)
        reduce(fracs[i], fracs[i + 1]);
    return 0;
}
