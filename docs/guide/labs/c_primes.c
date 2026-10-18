/* Lab: a C program on eMIPS (python3 -m emips cc c_primes.c).
 * Counts the primes below 100 with a sieve; c_primes.elf is the
 * prebuilt executable (tools/build_elf_fixtures.py). */
#include "emips.h"

#define N 100

static char composite[N];

int main(void)
{
    int count = 0;
    for (int i = 2; i < N; i++) {
        if (composite[i])
            continue;
        count++;
        for (int j = i * i; j < N; j += i)
            composite[j] = 1;
    }
    print_string("primes below 100: ");
    print_int(count);
    print_char('\n');
    return 0;
}
