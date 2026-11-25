/* MIPS by Example in eMIPS, episode 18 — GCD and the Sieve of Eratosthenes
 * (the C version of ep18_primes.asm). */
#include "emips.h"

static unsigned char flags[100];          /* flags[i] = 1: i is not prime */

unsigned gcd(unsigned a, unsigned b)      /* Euclid */
{
    while (b != 0) {
        unsigned r = a % b;
        a = b;
        b = r;
    }
    return a;
}

unsigned lcm(unsigned a, unsigned b)
{
    return a / gcd(a, b) * b;             /* divide first: no overflow */
}

void sieve(unsigned char *f, int n)
{
    f[0] = f[1] = 1;                      /* 0 and 1 are not prime */
    for (int i = 2; i * i < n; i++)
        if (!f[i])                        /* i is prime: cross out its multiples */
            for (int j = i * i; j < n; j += i)
                f[j] = 1;
}

static void print_call(const char *name, int a, int b, int r)
{
    print_string(name);
    print_int(a);
    print_string(", ");
    print_int(b);
    print_string(") = ");
    print_int(r);
    print_char('\n');
}

int main(void)
{
    int count = 0;

    print_call("gcd(", 1071, 462, gcd(1071, 462));
    print_call("lcm(", 12, 18, lcm(12, 18));
    sieve(flags, 100);
    print_string("primes below 100:");
    for (int i = 2; i < 100; i++)
        if (!flags[i]) {
            print_char(' ');
            print_int(i);
            count++;
        }
    print_char('\n');
    print_int(count);
    print_string(" primes\n");
    return 0;
}
