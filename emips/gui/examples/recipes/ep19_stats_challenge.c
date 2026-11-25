/* MIPS by Example in eMIPS, episode 19, the challenge solved (the C version
 * of ep19_stats_challenge.asm). */
#include "emips.h"

static const unsigned limits[2] = {10, 1000};

unsigned sum_mult(unsigned limit, unsigned k)   /* no loop */
{
    unsigned m = (limit - 1) / k;               /* how many multiples */
    return m * (m + 1) / 2 * k;
}

unsigned sum35(unsigned limit)
{
    return sum_mult(limit, 3) + sum_mult(limit, 5) - sum_mult(limit, 15);
}

void fizzbuzz(int n)
{
    int word = 0;
    if (n % 3 == 0) {
        print_string("Fizz");
        word = 1;
    }
    if (n % 5 == 0) {
        print_string("Buzz");
        word = 1;
    }
    if (!word)
        print_int(n);
}

int main(void)
{
    for (int i = 0; i < 2; i++) {
        print_string("below ");
        print_int(limits[i]);
        print_string(": ");
        print_int(sum35(limits[i]));
        print_char('\n');
    }
    for (int n = 1; n <= 15; n++) {
        fizzbuzz(n);
        print_char(n == 15 ? '\n' : ' ');
    }
    return 0;
}
